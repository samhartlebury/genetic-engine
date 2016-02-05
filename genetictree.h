#ifndef GENETICTREE_H
#define GENETICTREE_H

#include <QObject>
#include <QTreeWidgetItem>
#include <opencv/cv.hpp>
#include <opencv/cvaux.hpp>
#include <opencv/cxcore.hpp>
#include <opencv/highgui.h>

class GeneticTree : public QObject
{
    Q_OBJECT
public:
    explicit GeneticTree(QObject *parent = 0);

    struct GeneticTreeItem
    {
        enum Type {
            Constant,
            Matrix,
            Operator
        };

        enum Operations {
            Add,
            Divide,
            Multiply,
            Subtract
        };

        Type type;
        int depth;
        Operations operation;
        GeneticTreeItem *child1;
        GeneticTreeItem *child2;
        qreal constant;

        GeneticTreeItem& operator=(const GeneticTreeItem &source);
    };

    int depthOfTree();
    GeneticTree *breedWithTree(GeneticTree * const tree);
    void depth(const GeneticTreeItem *parent, int * const depth);
    uint maxInitialDepth;
    void generateTree();
    QTreeWidgetItem *generateUITree(); // Not to be used in console
    cv::Mat evaluateTree();
    void setMatrix(const QString &filePath);
    void setMatrix(cv::Mat input);
    GeneticTreeItem topItem;
    cv::Mat matrix;
    cv::Mat output;
    QTreeWidgetItem topUiItem;

    GeneticTree& operator=(const GeneticTree &source);
private:
    GeneticTreeItem &randomChild(int exclude = -1);
    void matrixOperation(qreal constant, GeneticTreeItem::Operations operation, bool reverseOrder = true);
    void bitwiseOperation(GeneticTreeItem::Operations operation, bool reverseOrder = true);
    void randomChildren(GeneticTreeItem * parent);
    void evaluateChildren(GeneticTreeItem const * parent);
    void uiChildren(GeneticTreeItem const * parent, QTreeWidgetItem * uiParent);
    QString typeToString(GeneticTreeItem::Type type);
    QString operatorToString(GeneticTree::GeneticTreeItem::Operations operation);
    GeneticTreeItem *getRandomChildOfTree(const GeneticTree *tree, int type = -1);
    void listOfChildren(QList<const GeneticTree::GeneticTreeItem *> &list, const GeneticTreeItem* parent);
    void removeTypeFromList(QList<const GeneticTreeItem *> &list, GeneticTreeItem::Type type);
};

#endif // GENETICTREE_H
