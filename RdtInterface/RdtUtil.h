#pragma once
#include <stdio.h>
#include <string>

#ifdef __ANDROID_API__
#include <android/log.h>
#include "jni.h"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, __func__, __VA_ARGS__))
#else
#define LOGD(...)  printf(__VA_ARGS__)
#endif
#define PRINTFLOW SequenceFlow s=SequenceFlow(__func__);

using namespace std;

class SequenceFlow {
	string mString;
public:
	SequenceFlow(string s);
	~SequenceFlow();;
};
