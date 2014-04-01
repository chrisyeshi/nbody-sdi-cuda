#include <QApplication>
#include "glwidget.h"

int main(int argc, char* argv[])
{
	QApplication app(argc, argv);
	GLWidget glWidget;
	glWidget.resize(800, 800);
	glWidget.show();

	return app.exec();
}