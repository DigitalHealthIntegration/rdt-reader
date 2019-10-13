/*
 * Copyright (C) 2016 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <string.h>
#include <jni.h>
#include <android/log.h>
#include <opencv2/core.hpp>
#include <vector>
#include "rdtReader.h"


#define LOG_TAG "IPRD"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__))


extern bool init();
extern AcceptanceStatus update(void *);

extern "C"
JNIEXPORT jobject JNICALL
Java_com_iprd_testapplication_MainActivity_update(JNIEnv *env, jobject thiz, jlong m) {
    char buff[100] = {0};
    LOGD("update %ld",(long)m);

    AcceptanceStatus acceptanceStatus;
    if (0 != m){
        acceptanceStatus = update(reinterpret_cast<void *>(m));
    }

    jclass ent_clazz = env->FindClass("com/iprd/testapplication/AcceptanceStatus");
    LOGD("ent_clazz %ld",(long)ent_clazz);
    if (!ent_clazz) return NULL;
#if  0
    jobject result = env->NewObject(ent_clazz,0,1,2,3,4,5,6,7,8,true);
    PRINTVAL("result","%ld",result);
    if (!result || env->ExceptionCheck() != JNI_FALSE) {
        env->ExceptionClear();
        return NULL;
    }
#else
    // Get the IDs of the constructor and the _myEntityType field
    jmethodID ent_init = env->GetMethodID(ent_clazz, "<init>", "()V");
    jobject result = env->NewObject(ent_clazz, ent_init);
    LOGD("result %ld",(long)result);

    if (!result || env->ExceptionCheck() != JNI_FALSE) {
        env->ExceptionClear();
        return NULL;
    }
    jfieldID fidmBrightness = env->GetFieldID(ent_clazz, "mBrightness", "S");
     if (!ent_init || !fidmBrightness) return NULL;
    //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
    env->SetShortField(result,fidmBrightness,acceptanceStatus.mBrightness);

    jfieldID fidmSharpness = env->GetFieldID(ent_clazz, "mSharpness", "S");

    if (!ent_init || !fidmSharpness) return NULL;
    env->SetShortField(result,fidmSharpness,acceptanceStatus.mSharpness);

    jfieldID fidmScale = env->GetFieldID(ent_clazz, "mScale", "S");
    if (!ent_init || !fidmScale) return NULL;
    env->SetShortField(result,fidmScale,acceptanceStatus.mScale);

    jfieldID fidmPerspectiveDistortion = env->GetFieldID(ent_clazz, "mPerspectiveDistortion", "S");
    if (!ent_init || !fidmPerspectiveDistortion) return NULL;
    env->SetShortField(result,fidmPerspectiveDistortion,acceptanceStatus.mPerspectiveDistortion);

    jfieldID fidmDisplacementX = env->GetFieldID(ent_clazz, "mDisplacementX", "S");
    if (!ent_init || !fidmDisplacementX) return NULL;
    env->SetShortField(result,fidmDisplacementX,acceptanceStatus.mDisplacementX);

    jfieldID fidmDisplacementY = env->GetFieldID(ent_clazz, "mDisplacementY", "S");
    if (!ent_init || !fidmDisplacementY) return NULL;
    env->SetShortField(result,fidmDisplacementY,acceptanceStatus.mDisplacementY);

    jfieldID fidmRDTFound = env->GetFieldID(ent_clazz, "mRDTFound", "Z");
    if (!ent_init || !fidmRDTFound) return NULL;
    env->SetBooleanField(result,fidmRDTFound, acceptanceStatus.mRDTFound);
#endif

    return  result;
}

extern "C"
JNIEXPORT jboolean JNICALL
Java_com_iprd_testapplication_MainActivity_init(JNIEnv *env, jobject thiz) {
    return init();
}