#include <cstdio> // Para utilizar archivos: fopen, fread, fwrite, gets...
#include <stdio.h>
#include <iostream> // Funciones cout y cin
#include <filesystem>
#include <experimental\filesystem>
#include <string>


#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2\reg\map.hpp>
#include "opencv2\reg\mappergradshift.hpp"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "exiv2\exiv2.hpp"

#include "Funciones varias.h"
///////////////////////////////////////////////////////////////////////////////////////////////////

using namespace std;
using namespace cv;
using namespace experimental::filesystem::v1;

///////////////////////////////////////////////////////////////////////////////////////////////////

int main()
{
	

	vector<string> bandas = { "GRE.TIF","RED.TIF","REG.TIF","NIR.TIF","RGB.JPG" };
			
			/*
			vector<vector<Point2f>> PuntosRojo = LeerPuntosDetectados("Listado_Puntos_RED.yml", 42);
			vector<vector<Point2f>> PuntosVerde = LeerPuntosDetectados("Listado_Puntos_GRE.yml", 42);

			Mat homografia = findHomography(PuntosVerde[0], PuntosRojo[0]); // Puntos a alinear, Puntos de referencia
			GuardarMat(homografia, "cosa.yml", "Homografia");
			Mat imagen1 = imread("set_calibrado_corregido/1GRE_k3.TIF", CV_LOAD_IMAGE_ANYDEPTH);
			Mat imagenfin;
			warpPerspective(imagen1, imagenfin, homografia, imagen1.size());
			imwrite("cosaGUAY.TIF", imagenfin);
			*/
			
	for (int i = 4; i <= 4; i++)
	{
		CorrigePezParrot("D:/calibracionParrot/", bandas[i], "correccion_Parrot/");
	}
	
	
	
	int num_k = 3;

	for (int i = 0; i <= 4; i++)
	{
	
		if (i < 4)
		{
			CalibraMono("D:/calibracion/", bandas[i], num_k, "deteccionesquina/");
			Mat matrizcam = LeerMatCamara("Matriz_Camara_" + bandas[i].substr(0, 3) +  "_k" + to_string(num_k) + ".yml");
			Mat distcoef = LeerMatDistorsion("Matriz_Distorsion_" + bandas[i].substr(0, 3) + "_k" + to_string(num_k) + ".yml", num_k);
			CorrigeImagenes(matrizcam, distcoef, bandas[i], "D:/calibracion/", "set_calibrado_corregido/");
		}
	
		if (i == 4)
		{
			CalibraRGB("D:/calibracion/", bandas[i], num_k, "deteccionesquina/");
			Mat matrizcam = LeerMatCamara("Matriz_Camara_" + bandas[i].substr(0, 3) + "_k" + to_string(num_k) + ".yml");
			Mat distcoef = LeerMatDistorsion("Matriz_Distorsion_" + bandas[i].substr(0, 3) + "_k" + to_string(num_k) + ".yml", num_k);
			CorrigeImagenesRGB(matrizcam, distcoef, bandas[i], "D:/calibracion/", "set_calibrado_corregido/");
		}
	
	
	
	}
	
	return(0);
	
}

/*
LOCALIZAR MIN Y MAX DE LA MATRIZ
double min, max;
cv::minMaxLoc(imagen, &min, &max);

/// Prueba calibración Agisoft
//cv::Mat matrizcam= Mat::eye(3, 3, CV_64F);
cv::Mat matrizcam2 = Mat(Size(3, 3), CV_64F);
matrizcam2.at<double>(0, 0) = 959.19;
matrizcam2.at<double>(0, 1) = 0.0;
matrizcam2.at<double>(0, 2) = 600.1268;
matrizcam2.at<double>(1, 0) = 0.0;
matrizcam2.at<double>(1, 1) = 959.19;
matrizcam2.at<double>(1, 2) = 415.4909;
matrizcam2.at<double>(2, 0) = 0.0;
matrizcam2.at<double>(2, 1) = 0.0;
matrizcam2.at<double>(2, 2) = 1.0;

//matrizcam.at<double>(1, 1) = 1.0; No condiciona nada fijar los valores de partida
//matrizcam.at<double>(0, 0) = 1.0;

//cv::Mat distcoef = Mat::zeros(8,1, CV_64F);
cv::Mat distcoef2 = Mat(Size(8, 1), CV_64F);
distcoef2.at<double>(0, 0) = -0.177077;
distcoef2.at<double>(0, 1) = 0.047056;
distcoef2.at<double>(0, 2) = 0.00268751;
distcoef2.at<double>(0, 3) = 0.0478248;
distcoef2.at<double>(0, 4) = -0.201056;
distcoef2.at<double>(0, 5) = 0.245166;
distcoef2.at<double>(0, 6) = 0.0;
distcoef2.at<double>(0, 7) = 0.0;
*/