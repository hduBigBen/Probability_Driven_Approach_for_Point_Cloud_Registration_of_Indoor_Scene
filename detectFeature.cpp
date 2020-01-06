#include "stdafx.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <ctime>
#include "slamBase.h"
// PCL ��
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/io/io.h>  
#include <pcl/visualization/cloud_viewer.h>  
#include <Eigen/SVD>  
#include <Eigen/Dense> 
// OpenCV �������ģ��
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp> // use this if you want to use SIFT or SURF
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// �����������
typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloud;
// ����ڲ�
//const double camera_factor = 1.0;
//const double camera_cx = 325.5;
//const double camera_cy = 253.5;
//const double camera_fx = 518.0;
//const double camera_fy = 519.0;
////use
//const double camera_factor = 1.00;//1000
//const double camera_cx = 325.5;// 3.2754244941119034e+02;//325.5
//const double camera_cy = 253.5;// 2.3965616633400465e+02;//253.5
//const double camera_fx = 518.0;// 5.9759790117450188e+02;//518.0
//const double camera_fy = 519.0;// 5.9765961112127485e+02;//519.0
//sun3D
const double camera_factor = 1;//1000
const double camera_cx = 3.20000000e+02;// 3.2754244941119034e+02;//325.5
const double camera_cy = 2.40000000e+02;// 2.3965616633400465e+02;//253.5
const double camera_fx = 5.70342205e+02;
const double camera_fy = 5.70342205e+02;
using namespace std;
using namespace cv;
using namespace Eigen;
using namespace Eigen::internal;
using namespace Eigen::Architecture;

//1.NYU���ݼ�
#define maxCorners 15
#define Error 0.03//����error+error
#define RightNumber 0.6
#define wid 561
#define hei 427
string path = R"(D:\recently\paper\experiments\NYU-2-5-ours)";
Mat rgb1 = imread(R"(D:\recently\paper\experiments\rgb2.png)", 1);
Mat rgb2 = imread(R"(D:\recently\paper\experiments\rgb5.png)", 1);
string namedepth1 = R"(D:\recently\paper\experiments\depth2.txt)";
string namedepth2 = R"(D:\recently\paper\experiments\depth5.txt)";
string namepointcloud1 = R"(D:\recently\paper\experiments\pointcloud2.pcd)";
string namepointcloud2 = R"(D:\recently\paper\experiments\pointcloud5.pcd)";

////2.ICL-NUIM���ݼ�
//#define maxCorners 15
//#define Error 20//����error*error
//#define RightNumber 0.5
//#define wid 640
//#define hei 480
//string path = R"(D:\recently\paper\experiments\ICL-NUIM-150-200-ours)";
//Mat rgb1 = imread(R"(D:\recently\paper\experiments\rgb150.jpg)", 1);
//Mat rgb2 = imread(R"(D:\recently\paper\experiments\rgb200.jpg)", 1);
//string namedepth1 = R"(D:\recently\paper\experiments\depth150.txt)";
//string namedepth2 = R"(D:\recently\paper\experiments\depth200.txt)";
//string namepointcloud1 = R"(D:\recently\paper\experiments\pointcloud150.pcd)";
//string namepointcloud2 = R"(D:\recently\paper\experiments\pointcloud200.pcd)";

////2.ICL-NUIM���ݼ�
//#define maxCorners 15
//#define Error 25
//#define RightNumber 0.4
//#define wid 640
//#define hei 480
//string path = R"(D:\recently\paper\experiments\ICL-NUIM-1140-1155-ours)";
//Mat rgb1 = imread(R"(D:\recently\paper\experiments\rgb1140.jpg)", 1);
//Mat rgb2 = imread(R"(D:\recently\paper\experiments\rgb1155.jpg)", 1);
//string namedepth1 = R"(D:\recently\paper\experiments\depth1140.txt)";
//string namedepth2 = R"(D:\recently\paper\experiments\depth1155.txt)";
//string namepointcloud1 = R"(D:\recently\paper\experiments\pointcloud1140.pcd)";
//string namepointcloud2 = R"(D:\recently\paper\experiments\pointcloud1155.pcd)";

////3. SUN3D���ݼ�
//#define maxCorners 15
//#define Error 30
//#define RightNumber 0.5
//#define wid 640
//#define hei 480
//string path = R"(D:\recently\paper\experiments\SUN3D-357-365-ours)";
//Mat rgb1 = imread(R"(D:\recently\paper\experiments\rgb357.png)", 1);
//Mat rgb2 = imread(R"(D:\recently\paper\experiments\rgb365.png)", 1);
//string namedepth1 = R"(D:\recently\paper\experiments\depth357.txt)";
//string namedepth2 = R"(D:\recently\paper\experiments\depth365.txt)";
//string namepointcloud1 = R"(D:\recently\paper\experiments\pointcloud357.pcd)";
//string namepointcloud2 = R"(D:\recently\paper\experiments\pointcloud365.pcd)";

Mat rgb_gray1, rgb_gray2;
vector<vector<vector<double>>>position1(3, vector<vector<double>>(hei, vector<double>(wid, 0)));//��һ��ͼ��ؼ�������
vector<vector<vector<double>>>position2(3, vector<vector<double>>(hei, vector<double>(wid, 0)));//�ڶ���ͼ��ؼ�������
double cornerdistance1[maxCorners][maxCorners];//��һ��ͼ��ؼ���֮��ľ���
double cornerdistance2[maxCorners][maxCorners];//�ڶ���ͼ��ؼ���֮��ľ���
vector<Point2f> corners1, corners2;//�ǵ�����
vector<int> corners1new, corners2new;//��һ�μ�����ĵ㼯��
vector<int> corners1newnew, corners2newnew;//�ڶ��μ�����ĵ㼯��
Matrix3f R(3, 3);
MatrixXf T;

void computeposition(Mat rgb, string namedepth,string namepointcloud, vector<vector<vector<double>>> &position)
{
	ifstream depth(namedepth);
	assert(depth.is_open());
	PointCloud::Ptr cloud(new PointCloud);

	for (int m = 0; m < wid; m++)//m�Ǻ����꣬����
		for (int n = 0; n < hei; n++)//n�������꣬����
		{
			string s;
			getline(depth, s);
			double d = atof(s.c_str());
			if (d == 0)
				continue;
			d = -(double(d) / camera_factor);
			PointT p;
			p.x = (m - camera_cx)*d / camera_fx;
			position[0][n][m] = p.x;
			p.y = (camera_cy - n)*d / camera_fy;
			position[1][n][m] = p.y;
			p.z = d;
			position[2][n][m] = p.z;
			//cout << m << " " << n << " " << position[0][n][m] << " " << position[1][n][m] << " " << position[2][n][m] << endl;
			p.b = rgb.ptr<uchar>(n)[m * 3];
			p.g = rgb.ptr<uchar>(n)[m * 3 + 1];
			p.r = rgb.ptr<uchar>(n)[m * 3 + 2];
			cloud->points.push_back(p);
		}
	cloud->height = 1;
	cloud->width = cloud->points.size();
	cloud->is_dense = false;
	pcl::io::savePCDFile(namepointcloud, *cloud);
	/*pcl::visualization::CloudViewer viewer("viewer");
	viewer.showCloud(cloud);
	system("pause");*/
}

void goodFeaturesToTrack_Demo()
{
	//��ԭͼ��һ�α���
	Mat copy1, copy2;
	copy1 = rgb1.clone();
	copy2 = rgb2.clone();
	// �ǵ���
	goodFeaturesToTrack(rgb_gray1, corners1, maxCorners, 0.01, 10, Mat(), 3, false, 0.04);
	goodFeaturesToTrack(rgb_gray2, corners2, maxCorners, 0.01, 10, Mat(), 3, false, 0.04);

	//������⵽�Ľǵ�
	//����ǵ������
	ofstream cornersposition(path + R"(\cornersposition.txt)");
	for (int i = 0; i < corners1.size(); i++)
	{
		int m = corners1[i].x;//m�Ǻ�����
		int n = corners1[i].y;//n��������
		cornersposition << i << "���㣺��" << m << "��" << n << "��  position:��" << position1[0][n][m] << "," << position1[1][n][m] << "," << position1[2][n][m] << "��  " << endl;
		circle(copy1, corners1[i], 5, Scalar(0, 255, 0), -1, 8, 0);

		string str = to_string(i);
		Point textpoint;
		textpoint.x = corners1[i].x - 5;
		textpoint.y = corners1[i].y + 5;
		putText(copy1, str, textpoint, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(0, 0, 255), 1);
	}
	cornersposition << endl << endl;
	for (int i = 0; i < corners2.size(); i++)
	{
		int m = corners2[i].x;//m�Ǻ�����
		int n = corners2[i].y;//n��������
		cornersposition << i << "���㣺��" << m << "��" << n << "��  position:��" << position2[0][n][m] << "," << position2[1][n][m] << "," << position2[2][n][m] << "��  " << endl;
		circle(copy2, corners2[i], 5, Scalar(0, 255, 0), -1, 8, 0);

		string str = to_string(i);
		Point textpoint;
		textpoint.x = corners2[i].x - 5;
		textpoint.y = corners2[i].y + 5;
		putText(copy2, str, textpoint, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(0, 0, 255), 1);
	}
	imshow("Image1", copy1);
	imshow("Image2", copy2);
}

void ComputeMatrix(MatrixXf Aa, MatrixXf Bb, double s1, double s2, double s3, double s4, double s5, double s6, int N)
{
	MatrixXf A, B;
	Vector3f centroidA, centroidB;
	centroidA[0] = s1 / N;
	centroidA[1] = s2 / N;
	centroidA[2] = s3 / N;
	centroidB[0] = s4 / N;
	centroidB[1] = s5 / N;
	centroidB[2] = s6 / N;
	A = Aa.block(0, 0, N, 3);
	B = Bb.block(0, 0, N, 3);

	for (int i = 0; i < N; i++)
	{
		A(i, 0) = A(i, 0) - centroidA[0];
		A(i, 1) = A(i, 1) - centroidA[1];
		A(i, 2) = A(i, 2) - centroidA[2];
		B(i, 0) = B(i, 0) - centroidB[0];
		B(i, 1) = B(i, 1) - centroidB[1];
		B(i, 2) = B(i, 2) - centroidB[2];
	}

	Matrix3f H(3, 3);
	H = A.transpose()*B;
	JacobiSVD<Eigen::MatrixXf> svd(H, ComputeThinU | ComputeThinV);
	Matrix3f V = svd.matrixV(), U = svd.matrixU();
	R = V*U.transpose();

	if (R.determinant() < 0)
	{
		cout << "Reflection detected" << endl;
		V(0, 3) = -1 * V(0, 3);
		V(1, 3) = -1 * V(1, 3);
		V(2, 3) = -1 * V(2, 3);
		R = V*U.transpose();
	}
	T = -R*centroidA + centroidB;
	//cout << R << endl;
	//cout << T << endl;
}

void ComputeMatch()
{
	//�ȼ������йؼ���֮��ľ���
	//ofstream cornerdistancetxt1(path + R"(\cornerdistance1.txt)");
	for (int i = 0; i < corners1.size(); i++)
	{
		for (int j = 0; j < corners1.size(); j++)
		{
			int x1 = corners1[i].x;//m�Ǻ�����
			int y1 = corners1[i].y;//n��������
			int x2 = corners1[j].x;//m�Ǻ�����
			int y2 = corners1[j].y;//n��������
			cornerdistance1[i][j] = sqrt(pow((position1[0][y1][x1] - position1[0][y2][x2]), 2) + pow((position1[1][y1][x1] - position1[1][y2][x2]), 2) + pow((position1[2][y1][x1] - position1[2][y2][x2]), 2));
			//cornerdistancetxt1 << left << setw(10) << cornerdistance1[i][j] << " ";// << cornerdistance1xyz[i][j][0] << "," << cornerdistance1xyz[i][j][1] << "," << cornerdistance1xyz[i][j][2] << ") " << endl;
		}
		//cornerdistancetxt1 << endl;
	}
	//ofstream cornerdistancetxt2(path + R"(\cornerdistance2.txt)");
	for (int i = 0; i < corners2.size(); i++)
	{
		//cornerdistancetxt2 << i << ":  ";
		for (int j = 0; j < corners2.size(); j++)
		{
			int x1 = corners2[i].x;//m�Ǻ�����
			int y1 = corners2[i].y;//n��������
			int x2 = corners2[j].x;//m�Ǻ�����
			int y2 = corners2[j].y;//n��������
			cornerdistance2[i][j] = sqrt(pow((position2[0][y1][x1] - position2[0][y2][x2]), 2) + pow((position2[1][y1][x1] - position2[1][y2][x2]), 2) + pow((position2[2][y1][x1] - position2[2][y2][x2]), 2));
			//cornerdistancetxt2 << left << setw(10) << cornerdistance2[i][j] << " ";// << cornerdistance2xyz[i][j][0] << "," << cornerdistance2xyz[i][j][1] << "," << cornerdistance2xyz[i][j][2] << ") " << endl;
		}
		//cornerdistancetxt2 << endl;
	}

	//����ͼƥ��
	MatrixXf Aa(10, 3), Bb(10, 3);//�ֱ𱣴��Ӧ�������
	double s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;//xyzֵ�ĺͣ�Ϊ��������
	int N = 0;//��Ӧ�ĵĸ���
	ofstream matchAB(path + R"(\match.txt)");
	matchAB << "����ͼ��Ӧ�����" << endl;
	for (int i1 = 0; i1 < corners1.size(); i1++)//���ѡȡ�ǵ�����
	{
		int maxMatch = 0;//���Ӧ����������ƥ����
		int matchB = -1;//������õ�Ķ�Ӧ��
		//����ýǵ�ƥ���������ұߵĽǵ�
		for (int i2 = 0; i2 < corners2.size(); i2++)//�ұ�ѡȡ�ǵ�����
		{
			int match = 0;//��ǰ�����ǵ��ƥ����
			//������߸ýǵ����ұ߸ýǵ��ƥ����
			for (int j1 = 0; j1 < corners1.size(); j1++)//��߽ǵ��������
			{
				if (cornerdistance1[i1][j1] != 0)//��ȥ���߾���Ϊ0��
				{
					double minError = 100;//������ߵ���С���
					int minNo = -1;//������߶�Ӧ���������
					//�жϸöξ����Ƿ���ƥ��
					for (int j2 = 0; j2 < corners2.size(); j2++)//�ұ߽ǵ��������
					{
						double t = fabs(cornerdistance1[i1][j1] - cornerdistance2[i2][j2]);//�����������ߵ����
						if ((t < minError) && (cornerdistance2[i2][j2] != 0))//С�ڵ�ǰ��С���
						{
							minError = t;//���浱ǰ����
							minNo = j2;
						}
					}
					if (minError < Error)//�����ǰ���С����ֵ����˵���Ƕ�Ӧ����
					{
						match++;
						//cout << "A: " << i1 << "-" << j1 << " B: " << i2 << "-" << minNo << endl;
					}
				}
			}
			//cout << "�� A" << i1 << "--B" << i2 << "��" << match << "��ƥ��" << endl;
			if (match > maxMatch)//����õ���ұ߸õ������ƥ��������
			{
				maxMatch = match;//��¼��ǰ��Ӧ��
				matchB = i2;
			}
		}
		if (maxMatch > RightNumber*corners1.size())//�������֮������ƥ������������Ҫ��������Ӧ��
		{
			//cout << "��¼��A" << i1 << "--B" << matchB << "��" << maxMatch << "������ƥ��" << endl;
			int x1 = corners1[i1].x;//m�Ǻ�����
			int y1 = corners1[i1].y;//n��������
			int x2 = corners2[matchB].x;//m�Ǻ�����
			int y2 = corners2[matchB].y;//n��������
			corners1new.push_back(i1);//�����Ӧ��������Ϊ�ڶ���ƥ�������
			corners2new.push_back(matchB);
			Aa(N, 0) = position1[0][y1][x1];
			s1 += position1[0][y1][x1];
			Aa(N, 1) = position1[1][y1][x1];
			s2 += position1[1][y1][x1];
			Aa(N, 2) = position1[2][y1][x1];
			s3 += position1[2][y1][x1];
			Bb(N, 0) = position2[0][y2][x2];
			s4 += position2[0][y2][x2];
			Bb(N, 1) = position2[1][y2][x2];
			s5 += position2[1][y2][x2];
			Bb(N, 2) = position2[2][y2][x2];
			s6 += position2[2][y2][x2];
			N++;
			matchAB << i1 << ": (" << position1[0][y1][x1] << "," << position1[1][y1][x1] << "," << position1[2][y1][x1] << ")--" << matchB << ": (" << position2[0][y2][x2] << "," << position2[1][y2][x2] << "," << position2[2][y2][x2] << ")  " << "��" << maxMatch << "������ƥ��" << endl;
		}
	}
	ComputeMatrix(Aa, Bb, s1, s2, s3, s4, s5, s6, N);

	////������һ�ζ�Ӧ��Ľ��
	//Mat copy1, copy2;
	//copy1 = rgb1.clone();
	//copy2 = rgb2.clone();
	////����ʣ��Ľǵ�
	//cout << "** Number of corners1new detected: " << corners1new.size() << endl;
	//cout << "** Number of corners2new detected: " << corners2new.size() << endl;
	//for (int i = 0; i < corners1new.size(); i++)
	//{
	//	Point2f p;
	//	p.x = corners1[corners1new[i]].x;
	//	p.y = corners1[corners1new[i]].y;
	//	circle(copy1, p, 4, Scalar(0, 0, 255), -1, 8, 0);
	//	//ͼ��ָ�룬Բ�ĵ����꣬Բ�뾶��Բ��ɫ��BGR����������ʾ������ϸ�̶ȣ�������ʾ�Ƿ���䣩���������ͣ�С����λ��
	//	string str = to_string(corners1new[i]);
	//	Point textpoint;
	//	textpoint.x = p.x - 5;
	//	textpoint.y = p.y + 5;
	//	putText(copy1, str, textpoint, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(0, 255, 0), 1);
	//}
	//for (int i = 0; i < corners2new.size(); i++)
	//{
	//	Point2f p;
	//	p.x = corners2[corners2new[i]].x;
	//	p.y = corners2[corners2new[i]].y;
	//	circle(copy2, p, 4, Scalar(0, 0, 255), -1, 8, 0);
	//	string str = to_string(corners2new[i]);
	//	Point textpoint;
	//	textpoint.x = p.x - 5;
	//	textpoint.y = p.y + 5;
	//	putText(copy2, str, textpoint, FONT_HERSHEY_TRIPLEX, 0.5, Scalar(0, 255, 0), 1);
	//}
	//namedWindow("Image1new", CV_WINDOW_AUTOSIZE);
	//namedWindow("Image2new", CV_WINDOW_AUTOSIZE);
	//imshow("Image1new", copy1);
	////imwrite(path + R"(\Imagecorner)" + to_string(no1) + R"(.png)", copy1);
	//imshow("Image2new", copy2);
	////imwrite(path + R"(\Imagecorner)" + to_string(no2) + R"(.png)", copy2);

	//�ڶ���ƥ��
	//MatrixXf Aa(10, 3), Bb(10, 3);//�ֱ𱣴��Ӧ�������
	//double s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;//xyzֵ�ĺͣ�Ϊ��������
	//int N = 0;//��Ӧ�ĵĸ���
	//matchAB << endl << endl << "�ڶ��ζ�Ӧ������" << endl;
	//for (int i1 = 0; i1 < corners1new.size(); i1++)//���ѡȡ�ǵ�����
	//{
	//	int maxMatch = 0;
	//	int matchB = -1;
	//	double minSumError = 100.000;
	//	//����ýǵ�ƥ���������ұߵĽǵ�
	//	for (int i2 = 0; i2 < corners2new.size(); i2++)//�ұ�ѡȡ�ǵ�����
	//	{
	//		int match = 0;//��ǰ�����ǵ��ƥ����
	//		double sumError = 0.000;//�����ǵ�ƥ��������
	//		//������߸ýǵ����ұ߸ýǵ��ƥ����
	//		for (int j1 = 0; j1 < corners1new.size(); j1++)//��߽ǵ��������
	//		{
	//			if (cornerdistance1[corners1new[i1]][corners1new[j1]] != 0)
	//			{
	//				double minError = 100;
	//				int minNo = -1;
	//				//�жϸöξ����Ƿ���ƥ��
	//				for (int j2 = 0; j2 < corners2new.size(); j2++)//�ұ߽ǵ��������
	//				{
	//					double t = abs(cornerdistance1[corners1new[i1]][corners1new[j1]] - cornerdistance2[corners2new[i2]][corners2new[j2]]);
	//					if ((t < minError) && (cornerdistance2[corners2new[i2]][corners2new[j2]] != 0))
	//					{
	//						minError = t;
	//						minNo = corners2new[j2];
	//					}
	//				}//for (int j2 = 0; j2 < maxCorners; j2++)
	//				if (minError < Error)
	//				{
	//					match++;
	//					cout << "A: " << corners1new[i1] << "-" << corners1new[j1] << " B: " << corners2new[i2] << "-" << minNo << endl;
	//					sumError = sumError + minError;
	//				}
	//			}
	//		}//for (int j1 = 0; j1 < maxCorners; j1++)
	//		cout << "�� A" << corners1new[i1] << "--B" << corners2new[i2] << "��" << match << "��ƥ�䣬�������" << sumError << endl;
	//		if (match > maxMatch)
	//		{
	//			maxMatch = match;
	//			matchB = corners2new[i2];
	//			minSumError = sumError;
	//		}
	//		if ((match == maxMatch) && (sumError <= minSumError))
	//		{
	//			maxMatch = match;
	//			matchB = corners2new[i2];
	//			minSumError = sumError;
	//		}
	//	}//for (int i2 = 0; i2 < maxCorners; i2++)
	//	if ((maxMatch >= RightNumber*corners1new.size()) && (minSumError <= SUMERROR))
	//	{
	//		//cout << "��¼��A" << i1 << "--B" << matchB << "��" << maxMatch << "������ƥ��" << endl;
	//		int x1 = corners1[corners1new[i1]].x;//m�Ǻ�����
	//		int y1 = corners1[corners1new[i1]].y;//n��������
	//		int x2 = corners2[matchB].x;//m�Ǻ�����
	//		int y2 = corners2[matchB].y;//n��������
	//		Aa(N, 0) = position1[0][y1][x1];
	//		s1 += position1[0][y1][x1];
	//		Aa(N, 1) = position1[1][y1][x1];
	//		s2 += position1[1][y1][x1];
	//		Aa(N, 2) = position1[2][y1][x1];
	//		s3 += position1[2][y1][x1];
	//		Bb(N, 0) = position2[0][y2][x2];
	//		s4 += position2[0][y2][x2];
	//		Bb(N, 1) = position2[1][y2][x2];
	//		s5 += position2[1][y2][x2];
	//		Bb(N, 2) = position2[2][y2][x2];
	//		s6 += position2[2][y2][x2];
	//		N++;
	//		
	//		matchAB << corners1new[i1] << ": (" << position1[0][y1][x1] << "," << position1[1][y1][x1] << "," << position1[2][y1][x1] << ")--"
	//		<< matchB << ": (" << position2[0][y2][x2] << "," << position2[1][y2][x2] << "," << position2[2][y2][x2] << ")  "
	//		<< "��" << maxMatch << "������ƥ�䣬�������" << minSumError << "ƽ����" << minSumError / maxMatch << endl;
	//		//matchAB << i1 << " " << matchB << " " << maxMatch << endl;
	//	}
	//}//for (int i1 = 0; i1 < maxCorners; i1++)
	//ComputeMatrix(Aa, Bb, s1, s2, s3, s4, s5, s6, N);

	//����ͼƥ��
	s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;//xyzֵ�ĺͣ�Ϊ��������
	N = 0;//��Ӧ�ĵĸ���
	matchAB << endl << endl << "����ͼ��Ӧ������" << endl;
	for (int i1 = 0; i1 < corners1.size(); i1++)//���ѡȡ�ǵ�����
	{
		int maxMatch = 0;
		int matchB = -1;
		//����ýǵ�ƥ���������ұߵĽǵ�
		for (int i2 = 0; i2 < corners2.size(); i2++)//�ұ�ѡȡ�ǵ�����
		{
			int match = 0;//��ǰ�����ǵ��ƥ����
			//������߸ýǵ����ұ߸ýǵ��ƥ����
			for (int j1 = 0; j1 < corners1.size(); j1++)//��߽ǵ��������
			{
				if (cornerdistance1[i1][j1] != 0)
				{
					double minError = 1000000;
					int minNo = -1;
					//�жϸöξ����Ƿ���ƥ��
					for (int j2 = 0; j2 < corners2.size(); j2++)//�ұ߽ǵ��������
					{
						double t = abs(cornerdistance1[i1][j1] - cornerdistance2[i2][j2]);
						int x1 = corners1[i1].x;
						int y1 = corners1[i1].y;
						int x2 = corners1[j1].x;
						int y2 = corners1[j1].y;
						Point3f point1t, point2t, point1, point2;
						point1t.x = position1[0][y1][x1];
						point1t.y = position1[1][y1][x1];
						point1t.z = position1[2][y1][x1];
						point1.x = R(0, 0) * point1t.x + R(0, 1) * point1t.y + R(0, 2) * point1t.z + T(0, 0);
						point1.y = R(1, 0) * point1t.x + R(1, 1) * point1t.y + R(1, 2) * point1t.z + T(1, 0);
						point1.z = R(2, 0) * point1t.x + R(2, 1) * point1t.y + R(2, 2) * point1t.z + T(2, 0);
						point2t.x = position1[0][y2][x2];
						point2t.y = position1[1][y2][x2];
						point2t.z = position1[2][y2][x2];
						point2.x = R(0, 0) * point2t.x + R(0, 1) * point2t.y + R(0, 2) * point2t.z + T(0, 0);
						point2.y = R(1, 0) * point2t.x + R(1, 1) * point2t.y + R(1, 2) * point2t.z + T(1, 0);
						point2.z = R(2, 0) * point2t.x + R(2, 1) * point2t.y + R(2, 2) * point2t.z + T(2, 0);
						double a1 = point1.x - point2.x;
						double b1 = point1.y - point2.y;
						double c1 = point1.z - point2.z;
						x1 = corners2[i2].x;//m�Ǻ�����
						y1 = corners2[i2].y;//n��������
						x2 = corners2[j2].x;//m�Ǻ�����
						y2 = corners2[j2].y;//n��������
						double a2 = position2[0][y1][x1] - position2[0][y2][x2];
						double b2 = position2[1][y1][x1] - position2[1][y2][x2];
						double c2 = position2[2][y1][x1] - position2[2][y2][x2];
						if ((t < minError) && (a1*a2 > 0) && (b1*b2 > 0) && (c1*c2 > 0) && (fabs(a1 - a2) < Error*Error) && (fabs(b1 - b2) < Error*Error) && (fabs(c1 - c2) < Error*Error) && (cornerdistance2[i2][j2] != 0))
						{
							minError = t;
							minNo = j2;
						}
					}//for (int j2 = 0; j2 < maxCorners; j2++)
					if (minError < Error*2)
					{
						match++;
						//cout << "A: " << i1 << "-" << j1 << " B: " << i2 << "-" << minNo << endl;
					}
				}
			}
			//cout << "�� A" << i1 << "--B" << i2 << "��" << match << "��ƥ��" << endl;
			if (match > maxMatch)
			{
				maxMatch = match;
				matchB = i2;
			}
		}
		if (maxMatch >= RightNumber*corners1.size())
		{
			//cout << "��¼��A" << i1 << "--B" << matchB << "��" << maxMatch << "������ƥ��" << endl;
			int x1 = corners1[i1].x;//m�Ǻ�����
			int y1 = corners1[i1].y;//n��������
			int x2 = corners2[matchB].x;//m�Ǻ�����
			int y2 = corners2[matchB].y;//n��������

			Aa(N, 0) = position1[0][y1][x1];
			s1 += position1[0][y1][x1];
			Aa(N, 1) = position1[1][y1][x1];
			s2 += position1[1][y1][x1];
			Aa(N, 2) = position1[2][y1][x1];
			s3 += position1[2][y1][x1];
			Bb(N, 0) = position2[0][y2][x2];
			s4 += position2[0][y2][x2];
			Bb(N, 1) = position2[1][y2][x2];
			s5 += position2[1][y2][x2];
			Bb(N, 2) = position2[2][y2][x2];
			s6 += position2[2][y2][x2];
			N++;

			matchAB << i1 << ": (" << position1[0][y1][x1] << "," << position1[1][y1][x1] << "," << position1[2][y1][x1] << ")--" << matchB << ": (" << position2[0][y2][x2] << "," << position2[1][y2][x2] << "," << position2[2][y2][x2] << ")  " << "��" << maxMatch << "������ƥ��" << endl;
		}
	}
	ComputeMatrix(Aa, Bb, s1, s2, s3, s4, s5, s6, N);

	////������ƥ��
	//s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;//xyzֵ�ĺͣ�Ϊ��������
	//N = 0;//��Ӧ�ĵĸ���
	//matchAB << endl << endl << "�����ζ�Ӧ������" << endl;
	//for (int i1 = 0; i1 < corners1new.size(); i1++)//���ѡȡ�ǵ�����
	//{
	//	int maxMatch = 0;
	//	int matchB = -1;
	//	double minSumError = 100.000;
	//	//����ýǵ�ƥ���������ұߵĽǵ�
	//	for (int i2 = 0; i2 < corners2new.size(); i2++)//�ұ�ѡȡ�ǵ�����
	//	{
	//		int match = 0;//��ǰ�����ǵ��ƥ����
	//		double sumError = 0.000;//�����ǵ�ƥ��������
	//								//������߸ýǵ����ұ߸ýǵ��ƥ����
	//		for (int j1 = 0; j1 < corners1new.size(); j1++)//��߽ǵ��������
	//		{
	//			if (cornerdistance1[corners1new[i1]][corners1new[j1]] != 0)
	//			{
	//				double minError = 100;
	//				int minNo = -1;
	//				//�жϸöξ����Ƿ���ƥ��
	//				for (int j2 = 0; j2 < corners2new.size(); j2++)//�ұ߽ǵ��������
	//				{
	//					double t = abs(cornerdistance1[corners1new[i1]][corners1new[j1]] - cornerdistance2[corners2new[i2]][corners2new[j2]]);
	//					int x1 = corners1[corners1new[i1]].x;
	//					int y1 = corners1[corners1new[i1]].y;
	//					int x2 = corners1[corners1new[j1]].x;
	//					int y2 = corners1[corners1new[j1]].y;
	//					double a1 = position1[0][y1][x1] - position1[0][y2][x2];
	//					double b1 = position1[1][y1][x1] - position1[1][y2][x2];
	//					double c1 = position1[2][y1][x1] - position1[2][y2][x2];
	//					x1 = corners2[corners2new[i2]].x;//m�Ǻ�����
	//					y1 = corners2[corners2new[i2]].y;//n��������
	//					x2 = corners2[corners2new[j2]].x;//m�Ǻ�����
	//					y2 = corners2[corners2new[j2]].y;//n��������
	//					Point3f point1t, point2t, point1, point2;
	//					point1t.x = position2[0][y1][x1] - T(0, 0);
	//					point1t.y = position2[1][y1][x1] - T(1, 0);
	//					point1t.z = position2[2][y1][x1] - T(2, 0);
	//					point1.x = R(0, 0) * point1t.x + R(1, 0) * point1t.y + R(2, 0) * point1t.z;//����
	//					point1.y = R(0, 1) * point1t.x + R(1, 1) * point1t.y + R(2, 1) * point1t.z;
	//					point1.z = R(0, 2) * point1t.x + R(1, 2) * point1t.y + R(2, 2) * point1t.z;
	//					point2t.x = position2[0][y2][x2] - T(0, 0);
	//					point2t.y = position2[1][y2][x2] - T(1, 0);
	//					point2t.z = position2[2][y2][x2] - T(2, 0);
	//					point2.x = R(0, 0) * point2t.x + R(1, 0) * point2t.y + R(2, 0) * point2t.z;//����
	//					point2.y = R(0, 1) * point2t.x + R(1, 1) * point2t.y + R(2, 1) * point2t.z;
	//					point2.z = R(0, 2) * point2t.x + R(1, 2) * point2t.y + R(2, 2) * point2t.z;
	//					double a2 = point1.x - point2.x;
	//					double b2 = point1.y - point2.y;
	//					double c2 = point1.z - point2.z;
	//					if ((t < minError) && (a1*a2 > 0) && (b1*b2 > 0) && (c1*c2 > 0) && (fabs(a1 - a2) < Error) && (fabs(b1 - b2) < Error) && (fabs(c1 - c2) < Error) && (cornerdistance2[corners2new[i2]][corners2new[j2]] != 0))
	//					{
	//						minError = t;
	//						minNo = corners2new[j2];
	//					}
	//				}//for (int j2 = 0; j2 < maxCorners; j2++)
	//				if (minError < Error)
	//				{
	//					match++;
	//					//cout << "A: " << corners1new[i1] << "-" << corners1new[j1] << " B: " << corners2new[i2] << "-" << minNo << endl;
	//					sumError = sumError + minError;
	//				}
	//			}
	//		}//for (int j1 = 0; j1 < maxCorners; j1++)
	//		 //cout << "�� A" << corners1new[i1] << "--B" << corners2new[i2] << "��" << match << "��ƥ�䣬�������" << sumError << endl;
	//		if (match > maxMatch)
	//		{
	//			maxMatch = match;
	//			matchB = corners2new[i2];
	//			minSumError = sumError;
	//		}
	//		if ((match == maxMatch) && (sumError <= minSumError))
	//		{
	//			maxMatch = match;
	//			matchB = corners2new[i2];
	//			minSumError = sumError;
	//		}
	//	}//for (int i2 = 0; i2 < maxCorners; i2++)
	//	if ((maxMatch >= RightNumber*corners1new.size()) && (minSumError <= SUMERROR))
	//	{
	//		//cout << "��¼��A" << i1 << "--B" << matchB << "��" << maxMatch << "������ƥ��" << endl;
	//		int x1 = corners1[corners1new[i1]].x;//m�Ǻ�����
	//		int y1 = corners1[corners1new[i1]].y;//n��������
	//		int x2 = corners2[matchB].x;//m�Ǻ�����
	//		int y2 = corners2[matchB].y;//n��������
	//		Aa(N, 0) = position1[0][y1][x1];
	//		s1 += position1[0][y1][x1];
	//		Aa(N, 1) = position1[1][y1][x1];
	//		s2 += position1[1][y1][x1];
	//		Aa(N, 2) = position1[2][y1][x1];
	//		s3 += position1[2][y1][x1];
	//		Bb(N, 0) = position2[0][y2][x2];
	//		s4 += position2[0][y2][x2];
	//		Bb(N, 1) = position2[1][y2][x2];
	//		s5 += position2[1][y2][x2];
	//		Bb(N, 2) = position2[2][y2][x2];
	//		s6 += position2[2][y2][x2];
	//		N++;
	//		matchAB << corners1new[i1] << ": (" << position1[0][y1][x1] << "," << position1[1][y1][x1] << "," << position1[2][y1][x1] << ")--"
	//			<< matchB << ": (" << position2[0][y2][x2] << "," << position2[1][y2][x2] << "," << position2[2][y2][x2] << ")  "
	//			<< "��" << maxMatch << "������ƥ�䣬�������" << minSumError << "ƽ����" << minSumError / maxMatch << endl;
	//		//matchAB << i1 << " " << matchB << " " << maxMatch << endl;
	//	}
	//}//for (int i1 = 0; i1 < maxCorners; i1++)
	//ComputeMatrix(Aa, Bb, s1, s2, s3, s4, s5, s6, N);
}

void test()
{
	PointCloud::Ptr cloud1(new PointCloud);
	PointCloud::Ptr cloud2(new PointCloud);
	pcl::io::loadPCDFile(namepointcloud1, *cloud1);
	pcl::io::loadPCDFile(namepointcloud2, *cloud2);
	for (int i = 0; i < cloud2->points.size(); ++i)
	{
		Point3f point;
		point.x = cloud2->points[i].x - T(0, 0);
		point.y = cloud2->points[i].y - T(1, 0);
		point.z = cloud2->points[i].z - T(2, 0);
		cloud2->points[i].x = R(0, 0) * point.x + R(1, 0) * point.y + R(2, 0) * point.z;//����
		cloud2->points[i].y = R(0, 1) * point.x + R(1, 1) * point.y + R(2, 1) * point.z;
		cloud2->points[i].z = R(0, 2) * point.x + R(1, 2) * point.y + R(2, 2) * point.z;
	}
	*cloud2 = *cloud1 + *cloud2;
	pcl::visualization::CloudViewer viewer("Cloud Viewer");
	viewer.showCloud(cloud2);
	//pcl::io::savePCDFile(nameoutpcd, *cloud2);
	system("PAUSE");
}

int main()
{
	clock_t start1, finish1;

	start1 = clock();
	computeposition(rgb1, namedepth1, namepointcloud1, position1);
	computeposition(rgb2, namedepth2, namepointcloud2, position2);
	//shi-Tomasi�ǵ���
	cvtColor(rgb1, rgb_gray1, CV_BGR2GRAY);
	cvtColor(rgb2, rgb_gray2, CV_BGR2GRAY);
	goodFeaturesToTrack_Demo();
	finish1 = clock();
	//ofstream time(path + R"(\time.txt)");
	//time << "preprocess:" << double(finish1 - start1) / CLOCKS_PER_SEC << "s" << endl;

	//namedWindow("Image1", CV_WINDOW_AUTOSIZE);
	//namedWindow("Image2", CV_WINDOW_AUTOSIZE);
	//����trackbar
	//int maxTrackbar = 100;
	//createTrackbar("MaxCorners:", "Image1", &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo);
	//createTrackbar("MaxCorners:", "Image2", &maxCorners, maxTrackbar, goodFeaturesToTrack_Demo);
	//imshow("Image", src);

	clock_t start, finish;
	start = clock();
	//�����Ӧ
	ComputeMatch();
	finish = clock();

	//cout << double(finish - start) / CLOCKS_PER_SEC << "s" << endl;
	//time << "registration:" << double(finish - start) / CLOCKS_PER_SEC << "s" << endl;
	//time << "total:" << double(finish - start + finish1 - start1) / CLOCKS_PER_SEC << "s" << endl;
	/*ofstream out(path + R"(\out.txt)");
	out << R(0, 0) << " " << R(0, 1) << " " << R(0, 2) << endl
		<< R(1, 0) << " " << R(1, 1) << " " << R(1, 2) << endl
		<< R(2, 0) << " " << R(2, 1) << " " << R(2, 2) << endl
		<< T(0, 0) << " " << T(1, 0) << " " << T(2, 0) << endl;*/
	waitKey(0);
	//test();
	return 0;
}