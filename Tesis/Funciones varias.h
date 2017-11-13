#pragma once
#include <opencv2/core/core.hpp>

using namespace cv;
using namespace std;

/// ALINEAR IMÁGENES CALIBRADAS
// Se utiliza para image registration con traslación. Imágenes en .TIF y 10 (16) bits.
Mat AlineaImg(Mat& imgbase, Mat& imgmovida);

/// GUARDAR MATRIZ/VECTOR/IMAGEN CUALQUIERA EN ARCHIVO DE TEXTO
// Guardar matriz de forma básica con los campos por defecto. Formato .yml
// ruta es la ruta con el nombre de archivo y el formato incluido.
// nombre_archivo es el nombre de la matriz dentro del archivo.
void GuardarMat(Mat& matriz, string ruta, string nombre_archivo);