#ifndef __RDTREADER__H
#define __RDTREADER__H
#include <stdint.h>

#define NOT_COMPUTED -100
#define TOO_HIGH  1
#define TOO_LOW -1
#define GOOD 0

class AcceptanceStatus {
public:
	AcceptanceStatus();
	~AcceptanceStatus();
	void setDefaultStatus(AcceptanceStatus *as);
	bool getRdtFound();
private:
	short mSharpness;
	short mScale;
	short mBrightness;
	short mPerspectiveDistortion;
	short mDisplacementX; //displacement of the RDT center.x from ideal
	short mDisplacementY; //displacement of the RDT center.y from ideal
	//	CvPoint mcenterOfRDT;
	bool mRDTFound;// was an RDT found by the coarse finder
};


bool init(void);
void update(void* ptr);
void term(void);

#endif __RDTREADER__H