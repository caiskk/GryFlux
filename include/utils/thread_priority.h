/*************************************************************************************************************************
 * Copyright 2024 Xidian619
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the “Software”), to deal
 *in the Software without restriction, including without limitation the rights
 *to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *copies of the Software, and to permit persons to whom the Software is
 *furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 *all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 *OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 *SOFTWARE.
 *************************************************************************************************************************/
#pragma once

#ifdef _WIN32
#include <Windows.h>
#else
#include <sched.h>
#include <pthread.h>
#include <sys/sysinfo.h>
#include <sys/resource.h>
#include <sys/file.h>
#endif

void SetThreadPriorityToMaxLevel() noexcept {
#ifdef _WIN32
    SetThreadPriority(GetCurrentProcess(), THREAD_PRIORITY_TIME_CRITICAL);
#else
    /* ps -eo state,uid,pid,ppid,rtprio,time,comm */
    struct sched_param param_;
    param_.sched_priority = sched_get_priority_max(SCHED_FIFO); // SCHED_RR
    pthread_setschedparam(pthread_self(), SCHED_FIFO, &param_);
#endif
}

bool WriteAllBytes(const char* path, const void* data, int length) noexcept {
    if (NULL == path || length < 0) {
        return false;
    }
 
    if (NULL == data && length != 0) {
        return false;
    }
 
    FILE* f = fopen(path, "wb+");
    if (NULL == f) {
        return false;
    }
 
    if (length > 0) {
        fwrite((char*)data, length, 1, f);   
    }
 
    fflush(f);
    fclose(f);
    return true;
}
 
void SetProcessPriorityToMaxLevel() noexcept {
#ifdef _WIN32
    SetPriorityClass(GetCurrentProcess(), REALTIME_PRIORITY_CLASS);
#else
    char path_[260];
    snprintf(path_, sizeof(path_), "/proc/%d/oom_adj", getpid());
 
    char level_[] = "-17";
    WriteAllBytes(path_, level_, sizeof(level_));
 
    /* Processo pai deve ter prioridade maior que os filhos. */
    setpriority(PRIO_PROCESS, 0, -20);
 
    /* ps -eo state,uid,pid,ppid,rtprio,time,comm */
    struct sched_param param_;
    param_.sched_priority = sched_get_priority_max(SCHED_FIFO); // SCHED_RR
    sched_setscheduler(getpid(), SCHED_RR, &param_);
#endif
}
