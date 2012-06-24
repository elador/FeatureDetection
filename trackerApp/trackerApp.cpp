// trackerApp.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace std;
using namespace cv;

int _tmain(int argc, _TCHAR* argv[])
{

	int device = 0;
	const string videoWindowName = "Live-video";

	string video;
	bool cam = true;
	bool realtime;

	int frameHeight;
	int frameWidth;

	bool running;

	//uchar* buffer;
	frameWidth = 640;
	frameHeight = 480;

	//buffer = new uchar[1];

	namedWindow(videoWindowName, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(videoWindowName.c_str(), 600, 50);


	running = true;

	VideoCapture capture;
	if (cam) {
		capture.open(device);
		if (!capture.isOpened()) {
			cerr << "Could not open stream from device " << device << endl;
			return 0;
		}
		if (!capture.set(CV_CAP_PROP_FRAME_WIDTH, frameWidth)
				|| !capture.set(CV_CAP_PROP_FRAME_HEIGHT, frameHeight))
			cerr << "Could not change resolution to " << frameWidth << "x" << frameHeight << endl;
		frameWidth = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
		frameHeight = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	} else {
		capture.open(video);
		if (!capture.isOpened()) {
			cerr << "Could not open video file '" << video << "'" << endl;
			return 0;
		}
		frameWidth = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_WIDTH));
		frameHeight = static_cast<int>(capture.get(CV_CAP_PROP_FRAME_HEIGHT));
	}


	bool first = true;
	Mat frame, image;
	size_t bufferSize = 0;

	timeval start, detStart, detEnd, frameStart, frameEnd;
	float allDetTimeS = 0;

	int frames = 0;
	gettimeofday(&start);
	cout.precision(2);

	cout << "Init done" << endl;
	cout << frameWidth << ", " << frameHeight << endl;

	string filename = "video/out_";
	std::ostringstream stm;


	char* fn_detFrontal = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_fd.mat";
	char* fn_regrSVR = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_ra.mat";
	char* fn_regrWVR = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_la.mat";

	char* fn_detRight= "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_re.mat";	// wvmr	
	char* fn_detLeft = "D:\\CloudStation\\libFD_patrik2011\\config\\fdetection\\fd_config_ffd_le.mat";	// wvml fd_hq64-scasferetmp+40 = subject looking LEFT

	FdImage *myimg;// = new FdImage();

	CascadeERT *ert = new CascadeERT();
	/*ert->wvm->load(fn_detFrontal);
	ert->svm->load(fn_detFrontal);
	ert->wvr->load(fn_regrWVR);
	ert->svr->load(fn_regrSVR);
	OverlapElimination *oe = new OverlapElimination();	// only init and set everything to 0
	oe->load(fn_detFrontal);
	*/
	ert->wvm->load(fn_detFrontal);
	ert->svm->load(fn_detFrontal);
	ert->oe->load(fn_detFrontal);
	ert->wvmr->load(fn_detRight);
	ert->svmr->load(fn_detRight);
	ert->oer->load(fn_detRight);
	ert->wvml->load(fn_detLeft);
	ert->svml->load(fn_detLeft);
	ert->oel->load(fn_detLeft);
	ert->wvr->load(fn_regrWVR);
	ert->svr->load(fn_regrSVR);

	Logger->global.img.drawScales = true;
	Logger->global.img.drawRegressorFoutAsText = true;

	while (running) {

		++frames;
		gettimeofday(&frameStart);
		capture >> frame;

		if (frame.empty()) {
			cerr << "Could not capture frame" << endl;
			running = false;
		} else {
			if (first) {
				first = false;
				image.create(frame.rows, frame.cols, frame.type());
			}
			cout << frame.rows << ", " << frame.cols << ", " << frame.type() << endl;
			gettimeofday(&detStart);

			myimg = new FdImage();
			myimg->load(&frame);

			/* RUN THE WHOLE TREE */
			/*ert->wvm->init_for_image(myimg);
			ert->wvm->extractToPyramids(myimg);
			std::vector<FdPatch*> candidates_wvm = ert->wvm->detect_on_image(myimg);
			
			candidates_wvm = oe->eliminate(candidates_wvm, ert->wvm->getIdentifier());
			candidates_wvm = ert->svm->detect_on_patchvec(candidates_wvm);
			cout << "NumPatches SVM: " << candidates_wvm.size() << endl;
			candidates_wvm = oe->exp_num_fp_elimination(candidates_wvm, ert->wvm->getIdentifier());
			ert->svr->detect_on_patchvec(candidates_wvm);*/
			ert->init_for_image(myimg);
			ert->detect_on_image(myimg);
			//	if(candidates_wvm.size()>0)
			//if(ert->candidates.size()>0)
			//	cout << "Angle SVR: " << ert->candidates[0]->fout[ert->svr->getIdentifier()] << endl;
			/* END TREE */

			gettimeofday(&detEnd);

			std::string best_s;
			float best_f;

			int vec_idx_best;
			float vec_all_best = 0.0f;
			//for(std::vector<FdPatch*>::iterator it = ert->candidates.begin(); it != ert->candidates.end(); ++it) {
			for(unsigned int i=0; i < ert->candidates.size(); ++i) {
				CertaintyMap::const_iterator cit = ert->candidates[i]->certainty.find(ert->svm->getIdentifier());
				if(cit != ert->candidates[i]->certainty.end()) {
					best_f = cit->second;
					best_s = ert->svm->getIdentifier();
				}
				CertaintyMap::const_iterator citl = ert->candidates[i]->certainty.find(ert->svml->getIdentifier());
				if(citl != ert->candidates[i]->certainty.end()) {
					if(citl->second > best_f) {
						best_f = citl->second;
						best_s = ert->svml->getIdentifier();
					}
				}
				CertaintyMap::const_iterator citr = ert->candidates[i]->certainty.find(ert->svmr->getIdentifier());
				if(citr != ert->candidates[i]->certainty.end()) {
					if(citr->second > best_f) {
						best_f = citr->second;
						best_s = ert->svmr->getIdentifier();
					}
				}
				if(best_f > vec_all_best) {
					vec_all_best = best_f;
					vec_idx_best = i;
				}
			}
			if(ert->candidates.size() > 1) {
				ert->candidates.erase(ert->candidates.begin()+vec_idx_best+1, ert->candidates.end());
			}
			if(ert->candidates.size() > 1) {
				ert->candidates.erase(ert->candidates.begin(), ert->candidates.begin()+vec_idx_best);
			}


			image = frame;
			Logger->drawBoxesWithAngleColor(image, ert->candidates, ert->wvr->getIdentifier());
			Logger->drawAllScaleBoxes(image, &myimg->pyramids, ert->wvr->getIdentifier(), 20, 20);
			
			imshow(videoWindowName, image);
			delete myimg;
			//Sleep(10);	// 10ms
			stm << filename << setfill('0') << setw(5) << frames << ".png";
			imwrite(stm.str(), image);
			//std::cout << stm.str() << std::endl;
			stm.str("");

			gettimeofday(&frameEnd);

			int itTimeMs = 1000 * (frameEnd.tv_sec - frameStart.tv_sec) + (frameEnd.tv_usec - frameStart.tv_usec) / 1000;
			int detTimeMs = 1000 * (detEnd.tv_sec - detStart.tv_sec) + (detEnd.tv_usec - detStart.tv_usec) / 1000;
			float allTimeS = 1.0 * (frameEnd.tv_sec - start.tv_sec) + 0.0000001 * (frameEnd.tv_usec - start.tv_usec);
			float allFps = frames / allTimeS;
			allDetTimeS += 0.001 * detTimeMs;
			float detFps = frames / allDetTimeS;
			cout << "frame: " << frames << "; time: " << itTimeMs << " ms (" << allFps << " fps); detection: " << detTimeMs << "ms (" << detFps << " fps)" << endl;

			int c = waitKey(10);
			if ((char) c == 'q') {
				running = false;
			}
		}
	}

	delete ert;
	//delete[] buffer;

	return 0;
}

