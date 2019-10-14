#ifndef __RDTREADER__H
#define __RDTREADER__H


#define NOT_COMPUTED -100
#define TOO_HIGH  1
#define TOO_LOW -1
#define GOOD 0

class usRect{
public:
	usRect() :x(0), y(0), width(0), height(0) {}
	unsigned short x, y;
	unsigned short width, height;
};

class AcceptanceStatus {
public:
	AcceptanceStatus();
	~AcceptanceStatus();
	bool getRdtFound();
	short mSharpness;
	short mScale;
	short mBrightness;
	short mPerspectiveDistortion;
	short mDisplacementX; //displacement of the RDT center.x from ideal
	short mDisplacementY; //displacement of the RDT center.y from ideal
	bool mRDTFound;// was an RDT found by the coarse finder
	usRect mBoundingBox;
private:
	void setDefaultStatus();
};

class Config{
public:
	Config() {
		mMaxScale = 1100;
		mMinScale = 700;
		mXMin = 100;
		mXMax = 500;
		mYMin = 50;
		mYMax = 650;
		mMinSharpness = 500.0f;
		mMaxBrightness = 210.0f;
		mMinBrightness = 160.0f;
	}

	short mMaxScale;
	short mMinScale;
	short mXMin;
	short mXMax;
	short mYMin;
	short mYMax;
	float mMinSharpness;
	float mMaxBrightness;
	float mMinBrightness;
};

#endif //__RDTREADER__H