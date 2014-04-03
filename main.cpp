#include <QApplication>
#include <cassert>
#include "glwidget.h"

int main(int argc, char* argv[])
{
	assert(argc == 2);
	QApplication app(argc, argv);
	GLWidget glWidget(atoi(argv[1]));
	glWidget.resize(800, 800);
	glWidget.show();

	return app.exec();
}