#ifndef __GLWIDGET_H__
#define __GLWIDGET_H__

#include <QGLWidget>
#include <QTimer>
#include <QElapsedTimer>
#include <QOpenGLBuffer>
#include "cuda/nbody.h"

class GLWidget : public QGLWidget
{
	Q_OBJECT
public:
	explicit GLWidget(int N, QWidget* parent = 0);

public slots:
	void advance();

protected:
    virtual void initializeGL();
    virtual void paintGL();
    virtual void resizeGL(int width, int height);
    virtual void keyPressEvent(QKeyEvent* event);

private:
	int N;
	NBody nbody;
	QTimer timer;
	QElapsedTimer elapsed;
	QOpenGLBuffer vbo, ibo;
};

#endif // __GLWIDGET_H__