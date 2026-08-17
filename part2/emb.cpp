#include <iostream>
#include <vector>
#include <random>
#include <iomanip>
#include <chrono>
#include <immintrin.h> 
#include <cstdlib>

using namespace std;
using namespace std::chrono;

const int embedding_table_size = 1000000;
const int embedding_dim = 128;
const int input_size = 720;
const int num_bags = 20;


int random_int(int range) {
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(0, range - 1);
    return dis(gen);
}

long long run_with_prefetching(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    
    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx   = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            int pf_index = j + 16;
            if (pf_index < end_idx) {
                const float* pf_ptr = &embedding_table[input[pf_index]*embedding_dim];
                _mm_prefetch((const char*)pf_ptr, _MM_HINT_T0);
            }

            // Now access the current row
            const float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) {
                bag_embedding[d] += data_ptr[d];
            }
        }

        output.push_back(bag_embedding);
    }
    //-------------------------------------------------------------------------------------------------------------------------------------------
    
    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();

    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx   = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            const float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; d += 16) {
                __m512 acc  = _mm512_loadu_ps(&bag_embedding[d]);
                __m512 vals = _mm512_loadu_ps(&data_ptr[d]);
                acc = _mm512_add_ps(acc, vals);
                _mm512_storeu_ps(&bag_embedding[d], acc);
            }
        }

        output.push_back(bag_embedding);
    }
    //-------------------------------------------------------------------------------------------------------------------------------------------

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH SIMD: " << duration.count() << " microseconds.";

    return duration.count();
}

long long run_with_prefetching_simd(const vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();

    //----------------------------------------------------- Write your code here ----------------------------------------------------------------
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx   = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            int pf_index = j + 16;
            if (pf_index < end_idx) {
                const float* pf_ptr = &embedding_table[input[pf_index] * embedding_dim];
                _mm_prefetch((const char*)pf_ptr, _MM_HINT_T0);
            }

            const float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; d += 16) {
                __m512 acc  = _mm512_loadu_ps(&bag_embedding[d]);
                __m512 vals = _mm512_loadu_ps(&data_ptr[d]);
                acc = _mm512_add_ps(acc, vals);
                _mm512_storeu_ps(&bag_embedding[d], acc);
            }
        }

        output.push_back(bag_embedding);
    }
    //-------------------------------------------------------------------------------------------------------------------------------------------

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITH software prefetching and SIMD: " << duration.count() << " microseconds.";

    return duration.count();
}


long long naive_emb(vector<float>& embedding_table, const vector<int>& input, const vector<int>& offsets) {

    auto start = high_resolution_clock::now();
    vector<vector<float>> output;

    for (size_t i = 0; i < offsets.size(); ++i) {
        int start_idx = offsets[i];
        int end_idx = (i + 1 < offsets.size()) ? offsets[i + 1] : input.size();

        vector<float> bag_embedding(embedding_dim, 0.0f);

        for (int j = start_idx; j < end_idx; ++j) {
            float* data_ptr = &embedding_table[input[j] * embedding_dim];
            for (int d = 0; d < embedding_dim; ++d) {
                bag_embedding[d] += data_ptr[d];
            }
        }

        output.push_back(bag_embedding);
    }

    auto end = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(end - start);
    cout << "\nTime WITHOUT software prefetching: " << duration.count() << " microseconds.";
    
    return duration.count();
}

int main() {
    // Prepare embedding table
    vector<float> embedding_table(embedding_table_size * embedding_dim);
    for (auto& val : embedding_table) {
        val = static_cast<float>(random_int(embedding_table_size));
    }

    // Input indices
    vector<int> input(input_size);
    for (auto& idx : input) {
        idx = random_int(embedding_table_size);
    }

    // Offsets
    vector<int> offsets;
    for (int i = 0; i < num_bags; ++i) {
        offsets.push_back((input_size * i) / num_bags);
    }

    // Run naive code 
    long long time_without_prefetch = naive_emb(embedding_table, input, offsets);
    
    // ---------- Flush Cache Before Running Prefetching ----------
    for (size_t i = 0; i < embedding_table.size(); i += 16) {
        _mm_clflush(&embedding_table[i]);
    }
    _mm_mfence();
    
    // Run emb with software prefetching 
    long long time_with_prefetch = run_with_prefetching(embedding_table, input, offsets);
    // Run emb with simd 
    long long time_with_simd = run_with_simd(embedding_table, input, offsets);
    // Run emb with software prefetching and simd
    long long time_with_prefetch_simd = run_with_prefetching_simd(embedding_table, input, offsets);
    

    // Compute speedup
    double speedup1 = static_cast<double>(time_without_prefetch) / time_with_prefetch;
    double speedup2 = static_cast<double>(time_without_prefetch) / time_with_simd;
    double speedup3 = static_cast<double>(time_without_prefetch) / time_with_prefetch_simd;
    cout << fixed << setprecision(3);
    cout << "\n\nSpeedup (with software prefetching) = " << speedup1 << "x\n";
    cout << "Speedup (with simd) = " << speedup2 << "x\n";
    cout << "Speedup (with software prefetching and simd) = " << speedup3 << "x\n";

    return 0;
}

