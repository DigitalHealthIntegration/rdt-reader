#include <stdio.h>
#include "RdtReader.h"
#ifndef EXTERN
#define EXTERN extern
#include "RdtProcessing.h"
#endif

int main(void) {
	
	RdtInterface*r = RdtInterface::getInstance();
	init();
	for (int i = 0; i < 10; i++) {
		update(NULL);
		printf("%d\n", r->getAcceptanceStatus().getRdtFound()?1:0);
		//sleep(100);
	}
//	term();
	getchar();
	return 0;
}
