
// Copyright 2024-present the vsag project
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "./utils.h"

#include <cstdint>
#include <unordered_set>

namespace vsag {

const static int64_t DEFAULT_WATCH_WINDOW_SIZE = 20;

SlowTaskTimer::SlowTaskTimer(const std::string& n, int64_t log_threshold_ms)
    : name(n), threshold(log_threshold_ms) {
    start = std::chrono::steady_clock::now();
}

SlowTaskTimer::~SlowTaskTimer() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    if (duration.count() > threshold) {
        if (duration.count() >= 1000) {
            logger::info("{0} cost {1:.3f}s", name, duration.count() / 1000);
        } else {
            logger::info("{0} cost {1:.3f}ms", name, duration.count());
        }
    }
}

Timer::Timer(double& ref) : ref_(ref) {
    start = std::chrono::steady_clock::now();
}

Timer::~Timer() {
    auto finish = std::chrono::steady_clock::now();
    std::chrono::duration<double, std::milli> duration = finish - start;
    ref_ = duration.count();
}

WindowResultQueue::WindowResultQueue() {
    queue_.resize(DEFAULT_WATCH_WINDOW_SIZE);
}

void
WindowResultQueue::Push(float value) {
    size_t window_size = queue_.size();
    queue_[count_ % window_size] = value;
    count_++;
}

float
WindowResultQueue::GetAvgResult() const {
    size_t statstic_num = std::min(count_, queue_.size());
    float result = 0;
    for (int i = 0; i < statstic_num; i++) {
        result += queue_[i];
    }
    return result / statstic_num;
}

std::string
format_map(const std::string& str, const std::unordered_map<std::string, std::string>& mappings) {
    std::string result = str;

    for (const auto& [key, value] : mappings) {
        size_t pos = result.find("{" + key + "}");
        while (pos != std::string::npos) {
            result.replace(pos, key.length() + 2, value);
            pos = result.find("{" + key + "}");
        }
    }
    return result;
}

}  // namespace vsag
