#include "RdtUtil.h"


SequenceFlow::SequenceFlow(string s) {
	mString = s;
	LOGD("Entering: %s\n", mString.c_str());
}

 SequenceFlow::~SequenceFlow() {
	LOGD("Exiting: %s\n", mString.c_str());
}
