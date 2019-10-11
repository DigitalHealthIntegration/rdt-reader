#include <stdio.h>
#include "RdtReader.h"
#ifndef EXTERN
#define EXTERN extern
#include "RdtProcessing.h"
#endif

int main(void) {
	
	init();
	for (int i = 0; i < 10; i++) {
		update(NULL);
		printf("%llu\n", mRdtStatus->mTimestamp);
		//sleep(100);
	}
	term();
	getchar();
	return 0;
}
