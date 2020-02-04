//
//  ObjectDetectionUtilsTest.m
//  ObjectDetectionUtilsTest
//
//  Created by developer on 28/01/20.
//  Copyright © 2020 IPRD. All rights reserved.
//
#include "../rdtcamera/ObjectDetectionUtil.hpp"
#import <XCTest/XCTest.h>
@interface ObjectDetectionUtilsTest : XCTestCase

@end

@implementation ObjectDetectionUtilsTest

- (void)setUp {
    // Put setup code here. This method is called before the invocation of each test method in the class.
}

- (void)tearDown {
    // Put teardown code here. This method is called after the invocation of each test method in the class.
}

- (void)testIdentity {
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    float s= 0.75f;
    CvPoint2D32f t(10.0f, 20.0f);
    CvPoint3D32f orientations(0,0,0);
    CvPoint2D32f est_scal_rot;
    XCTAssert(detect2(a, c, i,orientations, &est_scal_rot) == 0.0);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,1,0.000000001);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,0,0.000000001);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testTranslate {
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    float s= 0.75f;
    CvPoint2D32f t(10.0f, 20.0f);
    CvPoint3D32f orientations(0,0,0);
    CvPoint2D32f est_scal_rot;
    CvPoint2D32f a1=translate(a,t);
    CvPoint2D32f c1=translate(c,t);
    CvPoint2D32f i1=translate(i,t);
    
    XCTAssert(detect2(a1, c1, i1,orientations, &est_scal_rot) == 0.0);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,1,0.00001);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,0,0.00001);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testScale{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    float s= 0.75f;
    CvPoint2D32f t(10.0f, 20.0f);
    CvPoint3D32f orientations(0,0,0);
    CvPoint2D32f est_scal_rot;
    CvPoint2D32f a1=scale(a, s);
    CvPoint2D32f c1=scale(c,s);
    CvPoint2D32f i1=scale(i,s);

    XCTAssertEqualWithAccuracy(detect2(a1, c1, i1,orientations, &est_scal_rot), 0.0,3.0);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,0.75,0.00001);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,0,0.00001);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testRotation90{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    float s= 0.75f;
    CvPoint2D32f t(10.0f, 20.0f);
    CvPoint3D32f orientations(90,90,90);
    CvPoint2D32f est_scal_rot;
    CvPoint2D32f a1=swap(a);
    CvPoint2D32f c1=swap(c);
    CvPoint2D32f i1=swap(i);

    XCTAssertEqualWithAccuracy(detect2(a1, c1, i1,orientations, &est_scal_rot), 0.0,3.0);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,1.0,0.01);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,M_PI/2.0,0.01);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testRotation90WithScaleAndTranslate{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    float s= 0.75f;
    CvPoint2D32f t(10.0f, 20.0f);
    CvPoint3D32f orientations(90,90,90);
    CvPoint2D32f est_scal_rot;
    CvPoint2D32f a1=swap(scale(translate(a,t),s));
    CvPoint2D32f c1=swap(scale(translate(c,t),s));
    CvPoint2D32f i1=swap(scale(translate(i,t),s));

    XCTAssertEqualWithAccuracy(detect2(a1, c1, i1,orientations, &est_scal_rot), 0.0,3.0);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,0.75,0.01);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,M_PI/2.0,0.01);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testAffine{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    CvPoint3D32f orientations(30,30,30);
    CvPoint2D32f est_scal_rot;
    cv::Mat R =makeRMat(0.85,M_PI/6,CvPoint2D32f(14,23));
    CvPoint2D32f a1=warpPoint(a, R);
    CvPoint2D32f c1=warpPoint(c, R);
    CvPoint2D32f i1=warpPoint(i, R);

    XCTAssertEqualWithAccuracy(detect2(a1, c1, i1,orientations, &est_scal_rot), 0.0,3.0);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,0.85,0.01);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,M_PI/6.0,0.01);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testsmallAngleDisparity1{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    CvPoint3D32f orientations(0,22.5,22.5);
    CvPoint2D32f est_scal_rot;
    XCTAssertEqualWithAccuracy(detect2(a, c, i,orientations, &est_scal_rot), 0.0,0.001);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,1,0.01);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,0,0.01);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}

- (void)testsmallAngleDisparity2{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    CvPoint3D32f orientations(0,360-22.5,360-22.5);
    CvPoint2D32f est_scal_rot;
    XCTAssertEqualWithAccuracy(detect2(a, c, i,orientations, &est_scal_rot), 0.0,0.001);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,1,0.01);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,0,0.01);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testsmallAngleDisparity3{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    CvPoint3D32f orientations(0,22.5,360.0);
    CvPoint2D32f est_scal_rot;
    XCTAssertEqualWithAccuracy(detect2(a, c, i,orientations, &est_scal_rot), 0.0,0.001);
    XCTAssertEqualWithAccuracy(est_scal_rot.x,1,0.01);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,0,0.01);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testsmallAngleDisparity4{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    CvPoint3D32f orientations(67.5, 90, 90.0);
    CvPoint2D32f est_scal_rot;
    
    CvPoint2D32f a1=swap(a);
    CvPoint2D32f c1=swap(c);
    CvPoint2D32f i1=swap(i);

    XCTAssertEqualWithAccuracy(detect2(a1, c1, i1,orientations, &est_scal_rot), 0.0,3.0);

    XCTAssertEqualWithAccuracy(est_scal_rot.x,1,0.01);
    XCTAssertEqualWithAccuracy(est_scal_rot.y,M_PI/2,0.01);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testlargeAngleDisparity1{
    CvPoint2D32f a(152.0f, 30.0f);
    CvPoint2D32f c(746.0f, 30.0f);
    CvPoint2D32f i(874.0f, 30.0f);
    CvPoint3D32f orientations(0,22.5,45.0);
    CvPoint2D32f est_scal_rot;
    XCTAssertEqualWithAccuracy(detect2(a, c, i,orientations, &est_scal_rot), MAX_VALUE,0.001);

    // This is an example of a functional test case.
    // Use XCTAssert and related functions to verify your tests produce the correct results.
}
- (void)testPerformanceExample {
    // This is an example of a performance test case.
    [self measureBlock:^{
        // Put the code you want to measure the time of here.
    }];
}

@end
