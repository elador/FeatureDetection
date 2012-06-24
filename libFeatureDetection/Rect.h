#pragma once
class Rect
{
public:
	//Rect(void);
	~Rect(void);

	int left;
	int top;
	int right;
	int bottom;

    Rect(int l=0, int t=0, int r=0, int b=0): left(l), top(t), right(r), bottom(b) {}
    
	//inline Rect(const Rect& r): left(r.left), top(r.top), right(r.right), bottom(r.bottom) {}
    //inline bool PtInRect(POINT pt) { return ( (pt.x>=left) && (pt.x<=right) && (pt.y>=top) && (pt.x<=bottom) ); }
    //inline POINT TopLeft() { POINT pt={left,top}; return pt; }
    //inline POINT BottomRight() { POINT pt={right,bottom}; return pt; }
	//inline POINT CenterPoint() { POINT pt={(int)((left+right)/2),(int)((top+bottom)/2) }; return pt; }
	
	inline int Width() { return (right - left); }
	inline int Height() { return (bottom - top); }

    /*inline bool IntersectRect( const Rect& r1, const Rect& r2) {
    left=max(r1.left,r2.left); top=max(r1.top,r2.top);
    right=max(left,min(r1.right,r2.right)); bottom=max(top,min(r1.bottom,r2.bottom));
    return ((right>left) && (bottom>top));
    }
    inline bool IsRectEmpty() {
    return ( (right<=left) || (bottom<=top) );
    }*/
	bool operator== (const Rect& r) const { 
		return ( (r.left==left) && (r.right==right) && (r.top==top) && (r.bottom==bottom) );
	}

};

