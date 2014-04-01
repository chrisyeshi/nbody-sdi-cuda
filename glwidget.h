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
	explicit GLWidget(QWidget* parent = 0);

public slots:
	void advance();

protected:
    virtual void initializeGL();
    virtual void paintGL();
    virtual void resizeGL(int width, int height);
    virtual void keyPressEvent(QKeyEvent* event);

private:
	static const int N = 512;
	NBody nbody;
	QTimer timer;
	QElapsedTimer elapsed;
	QOpenGLBuffer vbo;
};

#endif // __GLWIDGET_H__