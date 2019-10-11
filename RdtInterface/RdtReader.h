#ifndef __RDTREADER__H
#define __RDTREADER__H
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

#define NOT_COMPUTED -100
#define TOO_HIGH  1
#define TOO_LOW -1
#define GOOD 0

#define LOGIT(x) {\
	printf("%s\n",x);\
}

typedef struct {
	short sharpness;
	short scale;
	short brightness;
	short perspectiveDistortion;
	short displacementX; //displacement of the RDT center.x from ideal
	short displacementY; //displacement of the RDT center.y from ideal
	bool bRDTFound;// was an RDT found by the coarse finder 
}AcceptanceStatus;

typedef struct {
	int64_t  mTimestamp;
	AcceptanceStatus mAcceptanceStatus;
	bool mIsAcceptable;
	float mSharpness;
	float mScale;
	float mBrightness;
	float mPerspectiveDistortion;
//	CvPoint mcenterOfRDT;
} RdtInterface;

void init();
RdtInterface update(void* ptr);
void term();

#endif __RDTREADER__H