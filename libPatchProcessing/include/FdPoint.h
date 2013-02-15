#pragma once
class FdPoint
{
public:
	FdPoint(void);
	~FdPoint(void);

	int x, y;			// position on input image
	int x_py, y_py;		// position in image pyramid
	int s;
};

