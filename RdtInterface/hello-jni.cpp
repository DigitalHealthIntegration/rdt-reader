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
#include "RdtProcessing.h"


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

    /* Get OtherClass */
    jfieldID fidOtherClass = env->GetFieldID(ent_clazz, "mBoundingBox", "Lcom/iprd/testapplication/usRect;");
    LOGD("fidOtherClass %ld",(long)fidOtherClass);

//TBD :: Set class memeber properly in class.
    jobject oVal = (env)->GetObjectField( result, fidOtherClass);
    LOGD("oVal %ld",(long)oVal);

    jclass clsOtherClass = (env)->GetObjectClass(oVal);
    LOGD("clsOtherClass %ld",(long)clsOtherClass);

    //    jobject boundingBox = (env)->GetObjectField(result, fidOtherClass);
//    LOGD("boundingBox %ld",(long)boundingBox);
//    jclass boundingBoxClass = (env)->GetObjectClass(boundingBox);
//    LOGD("boundingBoxClass %ld",(long)boundingBoxClass);

    {
        jclass usRectClass = env->FindClass("com/iprd/testapplication/usRect");
        LOGD("usRectClass %ld",(long)ent_clazz);
        if (!usRectClass) return NULL;
        // Get the IDs of the constructor and the _myEntityType field
        jmethodID usRectInit = env->GetMethodID(usRectClass, "<init>", "()V");
        jobject usRectObject = env->NewObject(usRectClass, usRectInit);
        LOGD("usRectObject %ld`",(long)usRectObject);

        if (!usRectObject || env->ExceptionCheck() != JNI_FALSE) {
            env->ExceptionClear();
            return NULL;
        }
        jfieldID fidX = env->GetFieldID(usRectClass, "x", "S");
        if (!usRectInit || !fidX) return NULL;
        //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
        env->SetShortField(usRectObject,fidX,acceptanceStatus.mBoundingBox.x);

        jfieldID fidY = env->GetFieldID(usRectClass, "y", "S");
        if (!usRectInit || !fidY) return NULL;
        //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
        env->SetShortField(usRectObject,fidY,acceptanceStatus.mBoundingBox.y);

        jfieldID fidWidth = env->GetFieldID(usRectClass, "width", "S");
        if (!usRectInit || !fidWidth) return NULL;
        //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
        env->SetShortField(usRectObject,fidX,acceptanceStatus.mBoundingBox.width);

        jfieldID fidHeight = env->GetFieldID(usRectClass, "height", "S");
        if (!usRectInit || !fidHeight) return NULL;
        //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
        env->SetShortField(usRectObject,fidX,acceptanceStatus.mBoundingBox.height);

//        jfieldID fidmRDTFound = env->GetObjectField(ent_clazz, "usRect");
//        LOGD("mBoundingBox %ld",(long)fidmRDTFound);
//
//        jfieldID fidmBoundingBox = env->GetFieldID(ent_clazz, "mBoundingBox", "Lcom/iprd/testapplication/usRect");
//        LOGD("fidmBoundingBox %ld",(long)fidmBoundingBox);
//
//        if (!ent_init || !fidmBoundingBox) return NULL;
//        env->SetObjectField(result, fidmBoundingBox, usRectObject);
    }
#endif

    return  result;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_iprd_testapplication_MainActivity_init(JNIEnv *env, jobject thiz, jobject c) {
    jclass cls = env->GetObjectClass(c);
    Config getConfVal;

    jfieldID fidmMaxScale = env->GetFieldID( cls, "mMaxScale", "S");
    jshort imMaxScale = env->GetShortField(c, fidmMaxScale);
    getConfVal.mMaxScale = imMaxScale;
    LOGD("mMaxScale: %d", imMaxScale);

    jfieldID fidmMinScale = env->GetFieldID( cls, "mMinScale", "S");
    jshort imMinScale = env->GetShortField(c, fidmMinScale);
    getConfVal.mMinScale = imMinScale;
    LOGD("mMinScale: %d", imMinScale);

    jfieldID fidmXMax = env->GetFieldID( cls, "mXMax", "S");
    jshort imXMax = env->GetShortField(c, fidmXMax);
    getConfVal.mXMax = imXMax;
    LOGD("mXMax: %d", imXMax);

    jfieldID fidmXMin = env->GetFieldID( cls, "mXMin", "S");
    jshort imXMin = env->GetShortField(c, fidmXMin);
    getConfVal.mXMin = imXMin;
    LOGD("mXMin: %d", imXMin);

    jfieldID fidmYMax = env->GetFieldID( cls, "mYMax", "S");
    jshort imYMax = env->GetShortField(c, fidmYMax);
    getConfVal.mYMax = imYMax;
    LOGD("mYMax: %d", imYMax);

    jfieldID fidmYMin = env->GetFieldID( cls, "mYMin", "S");
    jshort imYMin = env->GetShortField(c, fidmYMin);
    getConfVal.mYMin = imYMin;
    LOGD("mYMin: %d", imYMin);

    jfieldID fidmMaxBrightness = env->GetFieldID( cls, "mMaxBrightness", "F");
    jfloat imMaxBrightness = env->GetFloatField(c, fidmMaxBrightness);
    getConfVal.mMaxBrightness = imMaxBrightness;
    LOGD("mMaxBrightness: %f", imMaxBrightness);

    jfieldID fidmMinBrightness = env->GetFieldID( cls, "mMinBrightness", "F");
    jfloat imMinBrightness = env->GetFloatField(c, fidmMinBrightness);
    getConfVal.mMinBrightness = imMinBrightness;
    LOGD("mMinBrightness: %f", imMaxBrightness);

    jfieldID fidmMinSharpness = env->GetFieldID( cls, "mMinSharpness", "F");
    jfloat imMinSharpness = env->GetFloatField(c, fidmMinSharpness);
    getConfVal.mMinSharpness = imMinSharpness;
    LOGD("mMinSharpness: %f", imMinSharpness);

    RdtInterface::getInstance()->init(getConfVal);
}