#pragma once
class StdImage
{
public:
	StdImage(void);
	~StdImage(void);
	StdImage(unsigned char*, int, int, int=8);


	int w, h;		// image width, height
	int colordepth;
	unsigned char* data;	// as of now, this is mainly used!

	std::string filename;

	void writePNG(void);
	void writePNG(const std::string) const;

	unsigned char  pixelAt(int x, int y) const { return data[y*w+x]; }
	unsigned char& pixelAt(int x, int y)       { return data[y*w+x]; }
};
