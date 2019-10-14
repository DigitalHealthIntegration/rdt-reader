#include "RdtUtil.h"


SequenceFlow::SequenceFlow(string s) {
	mString = s;
	LOGD("Entering: %s", mString.c_str());
}

 SequenceFlow::~SequenceFlow() {
	LOGD("Exiting: %s", mString.c_str());
}
