/*
The sort-based approach is adapted from https://github.com/moderngpu/moderngpu
which has the following license:

> Copyright (c) 2016, Sean Baxter
> All rights reserved.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice, this
>    list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
> ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
> WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
> ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
> (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
> LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
> ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
> (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
> SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
>
> The views and conclusions contained in the software and documentation are those
> of the authors and should not be interpreted as representing official policies,
> either expressed or implied, of the FreeBSD Project.
*/

/* Sort-based join using moderngpu */

#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/kernel_scan.hxx>
#include <moderngpu/kernel_load_balance.hxx>

using namespace mgpu;

struct _join_bounds {
    mem_t<int> lower, upper;
};

template<typename launch_arg_t = empty_t,
  typename a_it, typename b_it, typename comp_t>
_join_bounds compute_join_bounds(a_it a, int a_count, b_it b, int b_count,
    comp_t comp, context_t& context) {

    mem_t<int> lower(a_count, context);
    mem_t<int> upper(a_count, context);
    sorted_search<bounds_lower, launch_arg_t>(a, a_count, b, b_count,
    lower.data(), comp, context);
    sorted_search<bounds_upper, launch_arg_t>(a, a_count, b, b_count,
    upper.data(), comp, context);

    // Prepare output
    _join_bounds bounds;
    lower.swap(bounds.lower);
    upper.swap(bounds.upper);
    return bounds;
}

template<typename launch_arg_t = empty_t>
mem_t<int> scan_join_bounds(const _join_bounds &bounds, int a_count, int b_count,
                            context_t &context, bool isInner,
                            int &out_join_count)
{
    // Compute output ranges by scanning upper - lower. Retrieve the reduction
    // of the scan, which specifies the size of the output array to allocate.
    mem_t<int> scanned_sizes(a_count, context);
    const int* lower_data = bounds.lower.data();
    const int* upper_data = bounds.upper.data();

    mem_t<int> count(1, context);

    if (isInner){
        transform_scan<int>([=]MGPU_DEVICE(int index) {
            return upper_data[index] - lower_data[index];
        }, a_count, scanned_sizes.data(), plus_t<int>(), count.data(), context);
    } else {
        transform_scan<int>([=]MGPU_DEVICE(int index) {
            auto out = upper_data[index] - lower_data[index];
            if ( upper_data[index] == lower_data[index] ){
                // for left-only keys, allocate a slot
                out += 1;
            }
            return out;
        }, a_count, scanned_sizes.data(), plus_t<int>(), count.data(), context);
    }

    // Prepare output
    out_join_count = from_mem(count)[0];
    return scanned_sizes;
}

template<typename launch_arg_t = empty_t>
mem_t<int> compute_joined_indices(const _join_bounds &bounds,
                                   const mem_t<int> &scanned_sizes,
                                   int a_count, int join_count,
                                   context_t &context,
                                   bool isInner, int append_count=0)
{
    // Allocate an int output array and use load-balancing search to compute
    // the join.

    const int* lower_data = bounds.lower.data();
    const int* upper_data = bounds.upper.data();

    // for outer join: allocate extra space for appending the right indices
    int output_npairs = join_count + append_count;
    mem_t<int> output(2 * output_npairs, context);
    int* output_data = output.data();

    if (isInner){
        // Use load-balancing search on the segments. The output is a pair with
        // a_index = seg and b_index = lower_data[seg] + rank.
        auto k = [=]MGPU_DEVICE(int index, int seg, int rank, const int *lower) {
            output_data[index] = seg;
            output_data[index + output_npairs] = lower[seg] + rank;
        };

        transform_lbs<launch_arg_t>(k, join_count, scanned_sizes.data(), a_count,
                                    context, lower_data);
    } else {
        // Use load-balancing search on the segments. The output is a pair with
        // a_index = seg
        // b_index = lower_data[seg] + rank { if lower_data[seg] != upper_data[seg] }
        //         = -1                     { otherwise }
        auto k = [=]MGPU_DEVICE(int index, int seg, int rank, tuple<int, int> lower_upper) {
            auto lower = get<0>(lower_upper);
            auto upper = get<1>(lower_upper);
            auto result = lower + rank;
            if ( lower == upper ) result = -1;
            output_data[index] = seg;
            output_data[index + output_npairs] = result;
        };
        transform_lbs<launch_arg_t>(k, join_count, scanned_sizes.data(), a_count,
                                    make_tuple(lower_data, upper_data), context);
    }
    return output;
}

template<typename launch_arg_t = empty_t, typename T>
void outer_join_append_right(T *output_data,
                             const mem_t<int> &matches,
                             int append_count, int join_count,
                             context_t &context) {
    int output_npairs = join_count + append_count;
    auto appender = [=]MGPU_DEVICE(int index, int seg, int rank) {
        output_data[index + join_count] = -1;
        output_data[index + join_count + output_npairs] = seg;
    };
    transform_lbs<launch_arg_t>(appender, append_count, matches.data(),
                                matches.size(), context);
}

template<typename launch_arg_t = empty_t,
         typename a_it, typename b_it, typename comp_t>
mem_t<int> outer_join_count_matches(a_it a, int a_count, b_it b, int b_count,
                                     comp_t comp, context_t &context,
                                     int &append_count)
{
    mem_t<int> matches(b_count, context);
    mem_t<int> matches_count(1, context);
    // Compute lower and upper bounds of b into a.
    mem_t<int> lower_rev(b_count, context);
    mem_t<int> upper_rev(b_count, context);
    sorted_search<bounds_lower, launch_arg_t>(
        b, b_count, a, a_count, lower_rev.data(), comp, context
    );
    sorted_search<bounds_upper, launch_arg_t>(
        b, b_count, a, a_count, upper_rev.data(), comp, context
    );

    const int* lower_rev_data = lower_rev.data();
    const int* upper_rev_data = upper_rev.data();
    transform_scan<int>([=]MGPU_DEVICE(int index){
        return upper_rev_data[index] == lower_rev_data[index];
    }, b_count, matches.data(), plus_t<int>(), matches_count.data(), context);

    // Prepare output
    append_count = from_mem(matches_count)[0];
    return matches;
}

template<typename launch_arg_t = empty_t,
         typename a_it, typename b_it, typename comp_t>
mem_t<int> inner_join(a_it a, int a_count, b_it b, int b_count,
                       comp_t comp, context_t& context)
{
    _join_bounds bounds = compute_join_bounds(a, a_count, b, b_count, comp, context);
    int join_count;
    mem_t<int> scanned_sizes = scan_join_bounds(bounds, a_count, b_count, context, true,
                                                join_count);
    mem_t<int> output = compute_joined_indices(bounds, scanned_sizes, a_count,
                                               join_count, context, true);
    return output;
}

template<typename launch_arg_t = empty_t,
         typename a_it, typename b_it, typename comp_t>
mem_t<int> left_join(a_it a, int a_count, b_it b, int b_count,
                      comp_t comp, context_t& context)
{
    _join_bounds bounds = compute_join_bounds(a, a_count, b, b_count, comp, context);
    int join_count;
    mem_t<int> scanned_sizes = scan_join_bounds(bounds, a_count, b_count, context, false,
                                                join_count);
    mem_t<int> output = compute_joined_indices(bounds, scanned_sizes, a_count,
                                               join_count, context, false, 0);
    return output;
}

template<typename launch_arg_t = empty_t,
  typename a_it, typename b_it, typename comp_t>
mem_t<int> outer_join(a_it a, int a_count, b_it b, int b_count,
                       comp_t comp, context_t& context)
{
    _join_bounds bounds = compute_join_bounds(a, a_count, b, b_count, comp,
                                              context);
    int join_count;
    mem_t<int> scanned_sizes = scan_join_bounds(bounds, a_count, b_count, context, false,
                                                join_count);
    int append_count;
    mem_t<int> matches = outer_join_count_matches(a, a_count, b, b_count,
                                                  comp, context, append_count );
    mem_t<int> output = compute_joined_indices(bounds, scanned_sizes, a_count,
                                               join_count, context, false, append_count);
    outer_join_append_right(output.data(), matches, append_count, join_count,
                            context);
    return output;
}

