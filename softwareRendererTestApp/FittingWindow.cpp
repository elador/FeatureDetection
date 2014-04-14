/*
 * FittingWindow.cpp
 *
 *  Created on: 04.04.2014
 *      Author: Patrik Huber
 */

#include "FittingWindow.hpp"

#include "imageio/LabeledImageSource.hpp"
#include "imageio/Landmark.hpp"
#include "imageio/LandmarkCollection.hpp"

#include "morphablemodel/OpenCVCameraEstimation.hpp"
#include "morphablemodel/AffineCameraEstimation.hpp"
#include "render/QOpenGLRenderer.hpp"
#include "render/SoftwareRenderer.hpp"
#include "render/MeshUtils.hpp"

#include "logging/LoggerFactory.hpp"

#include "boost/lexical_cast.hpp"

#include <chrono>

using imageio::Landmark;
using imageio::LandmarkCollection;
using logging::Logger;
using logging::LoggerFactory;
using cv::Mat;
using cv::Vec3f;
using cv::Point2f;
using cv::Scalar;
using boost::lexical_cast;


FittingWindow::FittingWindow(shared_ptr<imageio::LabeledImageSource> labeledImageSource, morphablemodel::MorphableModel morphableModel) : m_frame(0)/*, m_device(0)*/
{
	this->labeledImageSource = labeledImageSource;
	this->morphableModel = morphableModel;
}

void FittingWindow::initialize(QOpenGLContext* context)
{
	r = new render::QOpenGLRenderer(context);
	r->setViewport(width(), height(), devicePixelRatio());
}

void FittingWindow::render()
{
	// This function gets called by our subclass every time Qt is ready to render a frame
	// call r->setViewport before every render?
	float aspect = static_cast<float>(width()) / static_cast<float>(height());
	QMatrix4x4 matrix;
	matrix.ortho(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, 0.1f, 100.0f); // l r b t n f
	//matrix.ortho(-70.0f, 70.0f, -70.0f, 70.0f, 0.1f, 1000.0f);
	//matrix.perspective(30, aspect, 0.1, 100.0);
	matrix.translate(0, 0, -2);
	matrix.rotate(45.0f, 1.0f, 0.0f, 0.0f);
	//matrix.rotate(45.0f, 0.0f, 1.0f, 0.0f);
	//matrix.scale(1.0f/75.0f);
	//matrix.scale(0.003f);
	render::Mesh mesh = render::utils::MeshUtils::createCube();
	r->render(mesh, matrix);
	//r->render(morphableModel.getMean());


	// SOFTWARE RENDERER START
	Mat framebuffer = Mat::zeros(height(), width(), CV_8UC3);
	// Prepare:
	// this is only the method how to draw (e.g. make tris and draw them (i.e. duplicate vertices), but in the end we have a VertexShader that operates per vertex
	for (const auto& triIndices : mesh.tvi) {
		//For every triangle: Like OpenGL does it! (So actually, OpenGL duplicates the vertices as well, it's not using triangle indices, at least not the way how I draw)
		render::Triangle tri;
		tri.vertex[0] = mesh.vertex[triIndices[0]];
		tri.vertex[1] = mesh.vertex[triIndices[1]];
		tri.vertex[2] = mesh.vertex[triIndices[2]];
		// 
	}

	render::SoftwareRenderer swr;
	swr.enableTexturing(true);
	auto tex = std::make_shared<render::Texture>();
	tex->createFromFile("C:\\Users\\Patrik\\Documents\\GitHub\\isoRegistered3D_square.png");
	swr.setCurrentTexture(tex);
	auto swbuffs = swr.render(mesh, matrix);
	Mat swbuffc = swbuffs.first;
	Mat swbuffd = swbuffs.second;

	//cv::Mat framebuffer = cv::imread("C:\\Users\\Patrik\\Documents\\GitHub\\box_screenbuffer11.png");
	//cv::Mat textureMap = render::utils::MeshUtils::extractTexture(mesh, matrix, viewportWidth, viewportHeight, framebuffer);
	//cv::imwrite("C:\\Users\\Patrik\\Documents\\GitHub\\img_extracted11.png", textureMap);

	++m_frame;
}

void FittingWindow::fit()
{
	std::chrono::time_point<std::chrono::system_clock> start, end;
	Mat img;
	morphablemodel::OpenCVCameraEstimation epnpCameraEstimation(morphableModel); // todo: this can all go to only init once
	morphablemodel::AffineCameraEstimation affineCameraEstimation(morphableModel);
	vector<imageio::ModelLandmark> landmarks;
	Logger appLogger = Loggers->getLogger("fitter");

	//while (labeledImageSource->next()) {
	labeledImageSource->next();
	start = std::chrono::system_clock::now();
	appLogger.info("Starting to process " + labeledImageSource->getName().string());
	img = labeledImageSource->getImage();

	/*vector<cv::Rect> detectedFaces;
	faceCascade.detectMultiScale(img, detectedFaces, 1.2, 2, 0, cv::Size(50, 50));
	if (detectedFaces.empty()) {
	continue;
	}
	Mat output = img.clone();
	for (const auto& f : detectedFaces) {
	cv::rectangle(output, f, cv::Scalar(0.0f, 0.0f, 255.0f));
	}*/



	LandmarkCollection lms = labeledImageSource->getLandmarks();
	vector<shared_ptr<Landmark>> lmsv = lms.getLandmarks();
	landmarks.clear();
	Mat landmarksImage = img.clone(); // blue rect = the used landmarks
	for (const auto& lm : lmsv) {
		lm->draw(landmarksImage);
		//if (lm->getName() == "right.eye.corner_outer" || lm->getName() == "right.eye.corner_inner" || lm->getName() == "left.eye.corner_outer" || lm->getName() == "left.eye.corner_inner" || lm->getName() == "center.nose.tip" || lm->getName() == "right.lips.corner" || lm->getName() == "left.lips.corner") {
		landmarks.emplace_back(imageio::ModelLandmark(lm->getName(), lm->getPosition2D()));
		cv::rectangle(landmarksImage, cv::Point(cvRound(lm->getX() - 2.0f), cvRound(lm->getY() - 2.0f)), cv::Point(cvRound(lm->getX() + 2.0f), cvRound(lm->getY() + 2.0f)), cv::Scalar(255, 0, 0));
		//}
	}

	// Start affine camera estimation (Aldrian paper)
	Mat affineCamLandmarksProjectionImage = landmarksImage.clone(); // the affine LMs are currently not used (don't know how to render without z-vals)
	Mat affineCam = affineCameraEstimation.estimate(landmarks);
	for (const auto& lm : landmarks) {
		Vec3f tmp = morphableModel.getShapeModel().getMeanAtPoint(lm.getName());
		Mat p(4, 1, CV_32FC1);
		p.at<float>(0, 0) = tmp[0];
		p.at<float>(1, 0) = tmp[1];
		p.at<float>(2, 0) = tmp[2];
		p.at<float>(3, 0) = 1;
		Mat p2d = affineCam * p;
		Point2f pp(p2d.at<float>(0, 0), p2d.at<float>(1, 0)); // Todo: check
		cv::circle(affineCamLandmarksProjectionImage, pp, 4.0f, Scalar(0.0f, 255.0f, 0.0f));
	}
	// End Affine est.

	// Estimate the shape coefficients

	// $\hat{V} \in R^{3N\times m-1}$, subselect the rows of the eigenvector matrix $V$ associated with the $N$ feature points
	// And we insert a row of zeros after every third row, resulting in matrix $\hat{V}_h \in R^{4N\times m-1}$:
	Mat V_hat_h = Mat::zeros(4 * landmarks.size(), morphableModel.getShapeModel().getNumberOfPrincipalComponents(), CV_32FC1);
	int rowIndex = 0;
	for (const auto& lm : landmarks) {
		Mat basisRows = morphableModel.getShapeModel().getPcaBasis(lm.getName()); // getPcaBasis should return the not-normalized basis I think
		basisRows.copyTo(V_hat_h.rowRange(rowIndex, rowIndex + 3));
		rowIndex += 4; // replace 3 rows and skip the 4th one, it has all zeros
	}
	// Form a block diagonal matrix $P \in R^{3N\times 4N}$ in which the camera matrix C (P_Affine, affineCam) is placed on the diagonal:
	Mat P = Mat::zeros(3 * landmarks.size(), 4 * landmarks.size(), CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		Mat submatrixToReplace = P.colRange(4 * i, (4 * i) + 4).rowRange(3 * i, (3 * i) + 3);
		affineCam.copyTo(submatrixToReplace);
	}
	// The variances: We set the 3D and 2D variances to one static value for now. $sigma^2_2D = sqrt(1) + sqrt(3)^2 = 4$
	float sigma_2D = std::sqrt(4);
	Mat Sigma = Mat::zeros(3 * landmarks.size(), 3 * landmarks.size(), CV_32FC1);
	for (int i = 0; i < 3 * landmarks.size(); ++i) {
		Sigma.at<float>(i, i) = 1.0f / sigma_2D;
	}
	Mat Omega = Sigma.t() * Sigma;
	// The landmarks in matrix notation (in homogeneous coordinates), $3N\times 1$
	Mat y = Mat::ones(3 * landmarks.size(), 1, CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		y.at<float>(3 * i, 0) = landmarks[i].getX();
		y.at<float>((3 * i) + 1, 0) = landmarks[i].getY();
		// the position (3*i)+2 stays 1 (homogeneous coordinate)
	}
	// The mean, with an added homogeneous coordinate (x_1, y_1, z_1, 1, x_2, ...)^t
	Mat v_bar = Mat::ones(4 * landmarks.size(), 1, CV_32FC1);
	for (int i = 0; i < landmarks.size(); ++i) {
		Vec3f modelMean = morphableModel.getShapeModel().getMeanAtPoint(landmarks[i].getName());
		v_bar.at<float>(4 * i, 0) = modelMean[0];
		v_bar.at<float>((4 * i) + 1, 0) = modelMean[1];
		v_bar.at<float>((4 * i) + 2, 0) = modelMean[2];
		// the position (4*i)+3 stays 1 (homogeneous coordinate)
	}

	// Bring into standard regularised quadratic form with diagonal distance matrix Omega
	Mat A = P * V_hat_h;
	Mat b = P * v_bar - y;
	//Mat c_s; // The x, we solve for this! (the variance-normalized shape parameter vector, $c_s = [a_1/sigma_{s,1} , ..., a_m-1/sigma_{s,m-1}]^t$
	float lambda = 0.1f; // lambdaIn; //0.01f; // The weight of the regularisation
	int numShapePc = morphableModel.getShapeModel().getNumberOfPrincipalComponents();
	Mat AtOmegaA = A.t() * Omega * A;
	Mat AtOmegaAReg = AtOmegaA + lambda * Mat::eye(numShapePc, numShapePc, CV_32FC1);
	Mat AtOmegaARegInv = AtOmegaAReg.inv(/*cv::DECOMP_SVD*/);
	Mat AtOmegatb = A.t() * Omega.t() * b;
	Mat c_s = -AtOmegaARegInv * AtOmegatb;
	vector<float> fittedCoeffs(c_s);

	// End estimate the shape coefficients

	//std::shared_ptr<render::Mesh> meanMesh = std::make_shared<render::Mesh>(morphableModel.getMean());
	//render::Mesh::writeObj(*meanMesh.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\mean.obj");

	//const float aspect = (float)img.cols / (float)img.rows; // 640/480
	//render::Camera camera(Vec3f(0.0f, 0.0f, 0.0f), /*horizontalAngle*/0.0f*(CV_PI / 180.0f), /*verticalAngle*/0.0f*(CV_PI / 180.0f), render::Frustum(-1.0f*aspect, 1.0f*aspect, -1.0f, 1.0f, /*zNear*/-0.1f, /*zFar*/-100.0f));
	/*render::SoftwareRenderer r(img.cols, img.rows, camera); // 640, 480
	r.perspectiveDivision = render::SoftwareRenderer::PerspectiveDivision::None;
	r.doClippingInNDC = false;
	r.directToScreenTransform = true;
	r.doWindowTransform = false;
	r.setObjectToScreenTransform(shapemodels::AffineCameraEstimation::calculateFullMatrix(affineCam));
	r.draw(meshToDraw, nullptr);
	Mat buff = r.getImage();*/

	/*
	std::ofstream myfile;
	path coeffsFilename = outputPath / labeledImageSource->getName().stem();
	myfile.open(coeffsFilename.string() + ".txt");
	for (int i = 0; i < fittedCoeffs.size(); ++i) {
	myfile << fittedCoeffs[i] * std::sqrt(morphableModel.getShapeModel().getEigenvalue(i)) << std::endl;

	}
	myfile.close();
	*/

	//std::shared_ptr<render::Mesh> meshToDraw = std::make_shared<render::Mesh>(morphableModel.drawSample(fittedCoeffs, vector<float>(morphableModel.getColorModel().getNumberOfPrincipalComponents(), 0.0f)));
	//render::Mesh::writeObj(*meshToDraw.get(), "C:\\Users\\Patrik\\Documents\\GitHub\\fittedMesh.obj");

	//r.resetBuffers();
	//r.draw(meshToDraw, nullptr);
	// TODO: REPROJECT THE POINTS FROM THE C_S MODEL HERE AND SEE IF THE LMS REALLY GO FURTHER OUT OR JUST THE REST OF THE MESH

	//cv::imshow(windowName, img);
	//cv::waitKey(5);


	end = std::chrono::system_clock::now();
	int elapsed_mseconds = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	appLogger.info("Finished processing. Elapsed time: " + lexical_cast<string>(elapsed_mseconds)+"ms.\n");

	//}

}
