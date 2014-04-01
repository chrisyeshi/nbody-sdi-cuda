#include "glwidget.h"
#include <QKeyEvent>

GLWidget::GLWidget(QWidget* parent)
  : QGLWidget(parent), nbody(N)
{
	connect(&timer, SIGNAL(timeout()), this, SLOT(advance()));
}

//
//
// Public Slots
//
//

void GLWidget::advance()
{
	int msec = elapsed.elapsed();
	elapsed.restart();
	nbody.advance(float(msec) / 1000.f);
	updateGL();
}

//
//
// Virtual Methods from QGLWidget
//
//

void GLWidget::initializeGL()
{
	qglClearColor(QColor(0, 0, 0));
	glEnable(GL_POINT_SMOOTH);
	// glPointSize(15);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	timer.start(0);
	elapsed.start();
}

void GLWidget::paintGL()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0.0, 1.0, 0.0, 1.0, 0.0, 1.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	std::vector<float3> bodies = nbody.getBodies();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	qglColor(QColor(255, 255, 255, 225));
	glBegin(GL_POINTS);
	for (unsigned int i = 0; i < bodies.size(); ++i)
	{
		glVertex2f(bodies[i].x, bodies[i].y);
	}
	glEnd();
}

void GLWidget::resizeGL(int width, int height)
{
	glViewport(0, 0, width, height);
}

void GLWidget::keyPressEvent(QKeyEvent* event)
{
	if (event->key() == Qt::Key_Space)
	{
		nbody.advance(1.f);
		updateGL();
	}
}