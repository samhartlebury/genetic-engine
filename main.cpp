#include <QTimer>

#include "geneticengine.h"

int main(int argc, char *argv[])
{
    GeneticEngine engine(argc, argv);
    QTimer::singleShot(1, &engine, SLOT(start()));
    return engine.exec();
}

