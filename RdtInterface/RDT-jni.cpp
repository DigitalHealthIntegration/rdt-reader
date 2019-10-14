#include <string.h>
#include <opencv2/core.hpp>
#include <vector>
#include "RdtProcessing.h"
#include "RdtUtil.h"

extern "C"
JNIEXPORT jobject JNICALL
Java_com_iprd_rdtcamera_MainActivity_update(JNIEnv *env, jobject thiz, jlong m) {
    char buff[100] = {0};
//    LOGD("update %ld",(long)m);
    AcceptanceStatus acceptanceStatus;
    if (0 != m){
        acceptanceStatus = RdtInterface::getInstance()->process((void*)m);
    }
    jclass ent_clazz = env->FindClass("com/iprd/rdtcamera/AcceptanceStatus");
//    LOGD("ent_clazz %ld",(long)ent_clazz);
    if (!ent_clazz) return NULL;

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

    jfieldID fidX = env->GetFieldID(ent_clazz, "mBoundingBoxX", "S");
    if (!ent_init || !fidX) return NULL;
    //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
    env->SetShortField(result,fidX,acceptanceStatus.mBoundingBox.x);

    jfieldID fidY = env->GetFieldID(ent_clazz, "mBoundingBoxY", "S");
    if (!ent_init || !fidY) return NULL;
    //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
    env->SetShortField(result,fidY,acceptanceStatus.mBoundingBox.y);

    jfieldID fidWidth = env->GetFieldID(ent_clazz, "mBoundingBoxWidth", "S");
    if (!ent_init || !fidWidth) return NULL;
    //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
    env->SetShortField(result,fidWidth,acceptanceStatus.mBoundingBox.width);

    jfieldID fidHeight = env->GetFieldID(ent_clazz, "mBoundingBoxHeight", "S");
    if (!ent_init || !fidHeight) return NULL;
    //env->SetObjectField(result, fidmBrightness, reinterpret_cast<jobject>((short) 9));
    env->SetShortField(result,fidHeight,acceptanceStatus.mBoundingBox.height);

    return  result;
}

Config &getConfig(JNIEnv *env, jobject c, Config &getConfVal) {
    jclass cls = env->GetObjectClass(c);
    jfieldID fidmMaxScale = env->GetFieldID(cls, "mMaxScale", "S");
    jshort imMaxScale = env->GetShortField(c, fidmMaxScale);
    getConfVal.mMaxScale = imMaxScale;
    LOGD("mMaxScale: %d", imMaxScale);

    jfieldID fidmMinScale = env->GetFieldID(cls, "mMinScale", "S");
    jshort imMinScale = env->GetShortField(c, fidmMinScale);
    getConfVal.mMinScale = imMinScale;
    LOGD("mMinScale: %d", imMinScale);

    jfieldID fidmXMax = env->GetFieldID(cls, "mXMax", "S");
    jshort imXMax = env->GetShortField(c, fidmXMax);
    getConfVal.mXMax = imXMax;
    LOGD("mXMax: %d", imXMax);

    jfieldID fidmXMin = env->GetFieldID(cls, "mXMin", "S");
    jshort imXMin = env->GetShortField(c, fidmXMin);
    getConfVal.mXMin = imXMin;
    LOGD("mXMin: %d", imXMin);

    jfieldID fidmYMax = env->GetFieldID(cls, "mYMax", "S");
    jshort imYMax = env->GetShortField(c, fidmYMax);
    getConfVal.mYMax = imYMax;
    LOGD("mYMax: %d", imYMax);

    jfieldID fidmYMin = env->GetFieldID(cls, "mYMin", "S");
    jshort imYMin = env->GetShortField(c, fidmYMin);
    getConfVal.mYMin = imYMin;
    LOGD("mYMin: %d", imYMin);

    jfieldID fidmMaxBrightness = env->GetFieldID(cls, "mMaxBrightness", "F");
    jfloat imMaxBrightness = env->GetFloatField(c, fidmMaxBrightness);
    getConfVal.mMaxBrightness = imMaxBrightness;
    LOGD("mMaxBrightness: %f", imMaxBrightness);

    jfieldID fidmMinBrightness = env->GetFieldID(cls, "mMinBrightness", "F");
    jfloat imMinBrightness = env->GetFloatField(c, fidmMinBrightness);
    getConfVal.mMinBrightness = imMinBrightness;
    LOGD("mMinBrightness: %f", imMaxBrightness);

    jfieldID fidmMinSharpness = env->GetFieldID(cls, "mMinSharpness", "F");
    jfloat imMinSharpness = env->GetFloatField(c, fidmMinSharpness);
    getConfVal.mMinSharpness = imMinSharpness;
    LOGD("mMinSharpness: %f", imMinSharpness);
    return getConfVal;
}

extern "C"
JNIEXPORT void JNICALL
Java_com_iprd_rdtcamera_MainActivity_init(JNIEnv *env, jobject thiz, jobject c) {
    Config cVal;
    cVal = getConfig(env, c, cVal);
    RdtInterface::getInstance()->init(cVal);
}

extern "C"
JNIEXPORT void JNICALL
Java_com_iprd_rdtcamera_MainActivity_setConfig(JNIEnv *env, jobject thiz, jobject c) {
    Config cVal;
    cVal = getConfig(env, c, cVal);
    RdtInterface::getInstance()->setConfig(cVal);
}