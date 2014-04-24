#include "glwidget.h"
#include <cassert>
#include <QKeyEvent>

GLWidget::GLWidget(int N, QWidget* parent)
  : QGLWidget(parent), N(N), nbody(N),
    vbo(QOpenGLBuffer::VertexBuffer), ibo(QOpenGLBuffer::IndexBuffer)
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
	// std::cout << float(msec) / 1000.f << std::endl;
	nbody.advance(float(msec) / 1000.f);
	// nbody.advance(0.015);
	updateGL();
}

//
//
// Virtual Methods from QGLWidget
//
//

void GLWidget::initializeGL()
{
	// standard gl initialization
	qglClearColor(QColor(25, 25, 25));
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	// vertex buffer object for the points
	vbo.create();
	vbo.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	vbo.bind();
	vbo.allocate(N * sizeof(float3));
	vbo.release();
	// register vbo
	nbody.initBodies(vbo.bufferId());
	// ibo for triangles
	ibo.create();
	ibo.setUsagePattern(QOpenGLBuffer::DynamicDraw);
	// timer
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

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	qglColor(QColor(255, 255, 255, 225));

	// std::vector<float3> bodies = nbody.getBodies();
	// assert(bodies.size() == N);

	vbo.bind();
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(2, GL_FLOAT, sizeof(float3), 0);

	// ibo
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	std::vector<ushort3> trias = nbody.getTrias();
	ibo.bind();
	ibo.allocate(trias.data(), trias.size() * sizeof(ushort3));
	qglColor(QColor(240, 204, 127));
	glDrawElements(GL_TRIANGLES, trias.size() * 3, GL_UNSIGNED_SHORT, 0);
	ibo.release();
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	// qglColor(QColor(200, 70, 40));
	// glDrawArrays(GL_POINTS, 0, N);

	glDisableClientState(GL_VERTEX_ARRAY);
	vbo.release();

	// glBegin(GL_POINTS);
	// for (unsigned int i = 0; i < bodies.size(); ++i)
	// {
	// 	glVertex2f(bodies[i].x, bodies[i].y);
	// }
	// glEnd();
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