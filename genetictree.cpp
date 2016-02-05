#include "genetictree.h"
#include <QDebug>
#include <QDateTime>
#include <QThread>

GeneticTree::GeneticTree(QObject *parent) :
    QObject(parent),
    maxInitialDepth(2)
{

}

void GeneticTree::generateTree()
{
    //qsrand(1);
    qsrand(QDateTime::currentDateTime().toMSecsSinceEpoch()); // set random seed
    topItem.type = GeneticTreeItem::Operator;
    topItem.operation = GeneticTreeItem::Operations(qrand() % 4);
    topItem.depth = 0;
    randomChildren(&topItem);
}

int GeneticTree::depthOfTree()
{
    int treeDepth = 0;
    auto* top = &topItem;
    depth(top, &treeDepth);

    return treeDepth;
}

void GeneticTree::depth(const GeneticTreeItem *parent, int * const initialisedDepth)
{
    if (!parent)
        return;
    if (!parent->type)
        return;
    if (typeToString(parent->type) == "Undefined")
        return;

    *initialisedDepth = *initialisedDepth + 1;

    if (parent->type != GeneticTreeItem::Operator)
        return;

    auto* child1 = parent->child1;
    auto* child2 = parent->child2;

    depth(child1, initialisedDepth);
    int tempDepth = *initialisedDepth;
    depth(child2, initialisedDepth);

    if (tempDepth > *initialisedDepth)
        *initialisedDepth = tempDepth;
}

QTreeWidgetItem *GeneticTree::generateUITree()
{
    QTreeWidgetItem *uiItem = new QTreeWidgetItem;
    uiChildren(&topItem, uiItem);

    return uiItem;
}

cv::Mat GeneticTree::evaluateTree()
{
    evaluateChildren(&topItem);
    return output;
}

void GeneticTree::setMatrix(const QString &filePath)
{
    matrix = cv::imread(filePath.toStdString(), CV_LOAD_IMAGE_COLOR);
    matrix.convertTo(matrix, CV_32F);
    output = matrix.clone();
    output.convertTo(output, CV_32F);
}

void GeneticTree::setMatrix(cv::Mat input)
{
    matrix = input.clone();
    matrix.convertTo(matrix, CV_32F);
    output = matrix.clone();
    output.convertTo(output, CV_32F);
}

GeneticTree &GeneticTree::operator=(const GeneticTree &source)
{
    // Check for self-assignment
    if (this == &source)
        return *this;

    // Shallow copy source non-pointers
    maxInitialDepth = source.maxInitialDepth;
    topItem = source.topItem;
    matrix = source.matrix.clone();
    output = source.output.clone();
    topUiItem = source.topUiItem;

    return *this;
}

GeneticTree::GeneticTreeItem &GeneticTree::GeneticTreeItem::operator=(const GeneticTree::GeneticTreeItem &source)
{
    // Check for self-assignment
    if (this == &source)
        return *this;

    // Shallow copy source non-pointers
    constant = source.constant;
    depth = source.depth;
    operation = source.operation;
    type = source.type;

    // Deallocate our children
    //    if (child1)
    //        delete child1;
    //    if (child2)
    //        delete child2;

    // Deep copy source children
    if (source.child1 && source.child2) {
        // Allocate memory and copy
        child1 = new GeneticTreeItem;
        child1 = source.child1;
        child2 = new GeneticTreeItem;
        child2 = source.child2;
    } else {
        child1 = 0;
        child2 = 0;
    }

    return *this;
}

GeneticTree* GeneticTree::breedWithTree(GeneticTree  * const tree)
{
    GeneticTree *child = new GeneticTree;
    child->topItem = tree->topItem;
    child->matrix = tree->matrix.clone();
    GeneticTreeItem *randomChildOfChild = getRandomChildOfTree(child);
    GeneticTreeItem *randomChildOfThis = getRandomChildOfTree(this, randomChildOfChild->type);

    *randomChildOfChild = *randomChildOfThis;

    // Returning unevaluated child
    return child;
}

GeneticTree::GeneticTreeItem* GeneticTree::getRandomChildOfTree(GeneticTree const * tree, int type)
{
    QList<const GeneticTreeItem*> childList;

    listOfChildren(childList, &tree->topItem);

    //GeneticTreeItem::Type itemType = static_cast<GeneticTreeItem::Type>(type);

    // For now, only allow operator swapping
    removeTypeFromList(childList, GeneticTreeItem::Constant);
    removeTypeFromList(childList, GeneticTreeItem::Matrix);

    if (childList.size() == 0)
        Q_ASSERT(false);

    int randomNum = (qrand() % childList.size());

    return const_cast<GeneticTreeItem*>(childList[randomNum]);
}

void GeneticTree::removeTypeFromList(QList<const GeneticTreeItem*> &list, GeneticTreeItem::Type type)
{
    QMutableListIterator<const GeneticTreeItem*> i(list);
    while (i.hasNext()) {
        const auto& item = i.next();
        if (item->type == type)
            i.remove();
    }
}

void GeneticTree::listOfChildren(QList<const GeneticTree::GeneticTreeItem*> &list, const GeneticTreeItem* parent)
{
    list.append(parent);

    if (parent->type != GeneticTreeItem::Operator)
        return;

    listOfChildren(list, parent->child1);
    listOfChildren(list, parent->child2);
}

void GeneticTree::evaluateChildren(GeneticTreeItem const * parent)
{
    if (parent->type != GeneticTreeItem::Operator)
        return;

    evaluateChildren(parent->child1);
    evaluateChildren(parent->child2);

    if (parent->child1->type == GeneticTreeItem::Constant && parent->child2->type == GeneticTreeItem::Matrix)
        matrixOperation(parent->child1->constant, parent->operation, true);
    else if (parent->child1->type == GeneticTreeItem::Matrix && parent->child2->type == GeneticTreeItem::Constant)
        matrixOperation(parent->child2->constant, parent->operation, false);
    else if (parent->child1->type == GeneticTreeItem::Constant && parent->child2->type == GeneticTreeItem::Operator)
        matrixOperation(parent->child1->constant, parent->operation, true);
    else if (parent->child1->type == GeneticTreeItem::Operator && parent->child2->type == GeneticTreeItem::Constant)
        matrixOperation(parent->child2->constant, parent->operation, false);
    else if (parent->child1->type == GeneticTreeItem::Matrix && parent->child2->type == GeneticTreeItem::Operator)
        bitwiseOperation(parent->operation, true);
    else if (parent->child1->type == GeneticTreeItem::Operator && parent->child2->type == GeneticTreeItem::Matrix)
        bitwiseOperation(parent->operation, false);

    return;
}

void GeneticTree::uiChildren(GeneticTree::GeneticTreeItem const * parent, QTreeWidgetItem *uiParent)
{
    if (parent->type == GeneticTreeItem::Operator) {
        uiParent->setText(0, operatorToString(parent->operation));
    } else if (parent->type == GeneticTreeItem::Constant) {
        uiParent->setText(0, QString::number(parent->constant));
        return;
    } else {
        uiParent->setText(0, typeToString(parent->type));
        return;
    }

    QTreeWidgetItem *uiChild1 = new QTreeWidgetItem;
    uiParent->addChild(uiChild1);
    uiChildren(parent->child1, uiParent->child(0));

    QTreeWidgetItem *uiChild2 = new QTreeWidgetItem;
    uiParent->addChild(uiChild2);
    uiChildren(parent->child2, uiParent->child(1));
}

QString GeneticTree::typeToString(GeneticTree::GeneticTreeItem::Type type)
{
    switch (type) {
    case GeneticTreeItem::Operator: return "Operator";
    case GeneticTreeItem::Matrix: return "Matrix";
    case GeneticTreeItem::Constant: return "Constant";
    default: return "Undefined";
    }
}

QString GeneticTree::operatorToString(GeneticTree::GeneticTreeItem::Operations operation)
{
    switch (operation) {
    case GeneticTreeItem::Add: return "Add";
    case GeneticTreeItem::Divide: return "Divide";
    case GeneticTreeItem::Multiply: return "Multiply";
    case GeneticTreeItem::Subtract: return "Subtract";
    default: return "Undefined operation";
    }
}

void GeneticTree::randomChildren(GeneticTreeItem * parent)
{
    if (!parent)
        return;

    if (parent->type != GeneticTreeItem::Operator)
        return;

    int maxDepthExclusionCode = 2; // Default set to exclude operators
    uint depth = parent->depth + 1;

    if (depth < maxInitialDepth - 1) // Parsimony pressure
        maxDepthExclusionCode = -1; // Set to no exlcusion

    parent->child1 = &randomChild(maxDepthExclusionCode);

    parent->child1->depth = depth;

    int child1Type = parent->child1->type;

    if (child1Type == GeneticTreeItem::Operator) {
        randomChildren(parent->child1);
        parent->child2 = &randomChild(maxDepthExclusionCode);

    } else if (maxDepthExclusionCode == 2) { // Reached max depth so stop growth
        parent->child2 = &randomChild(2);

        parent->child2->type = static_cast<GeneticTreeItem::Type>(!parent->child1->type); // Rule, cant be same type at max depth
    } else {
        parent->child2 = &randomChild(child1Type);

    }
    parent->child2->depth = depth;
    if (parent->child2->type == GeneticTreeItem::Operator)
        randomChildren(parent->child2);
}

GeneticTree::GeneticTreeItem& GeneticTree::randomChild(int exclude)
{
    GeneticTreeItem *item = new GeneticTreeItem;

    switch (exclude) {
    case -1: item->type = GeneticTreeItem::Type(qrand() % 3); break; // Exclude none
    case  0: item->type = GeneticTreeItem::Type((qrand() % 2) + 1); break; // Exclude constants
    case  1: item->type = GeneticTreeItem::Type((qrand() % 2) * 2); break; // Exclude matrices
    case  2: item->type = GeneticTreeItem::Type(qrand() % 2); break; // Exclude operators
    }

    item->operation = GeneticTreeItem::Operations(qrand() % 4);
    item->constant = qreal(qrand() % 1000) / 1000;

    return *item;
}

void GeneticTree::matrixOperation(qreal constant, GeneticTree::GeneticTreeItem::Operations operation, bool reverseOrder)
{
    switch (static_cast<GeneticTreeItem::Operations>(operation)) {
    case GeneticTreeItem::Add: output = matrix + constant; return;
    case GeneticTreeItem::Divide: reverseOrder ? output = constant / matrix : output = matrix / constant; return;
    case GeneticTreeItem::Multiply: output = matrix * constant; return;
    case GeneticTreeItem::Subtract: reverseOrder ? output = constant - matrix : output = matrix - constant; return;
    default: Q_ASSERT(false);
    }
}

void GeneticTree::bitwiseOperation(GeneticTree::GeneticTreeItem::Operations operation, bool reverseOrder)
{
    switch (static_cast<GeneticTreeItem::Operations>(operation)) {
    case GeneticTreeItem::Add: output = output + matrix; return;
    case GeneticTreeItem::Divide: reverseOrder ? output = output / matrix : output = matrix / output; return;
    case GeneticTreeItem::Multiply: output = matrix * output; return;
    case GeneticTreeItem::Subtract: reverseOrder ? output = output - matrix : output = matrix - output; return;
    default: Q_ASSERT(false);
    }
}
