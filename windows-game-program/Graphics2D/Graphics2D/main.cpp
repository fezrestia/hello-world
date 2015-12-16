#include "GameLib/Framework.h"

#include <iostream>
#include <stdlib.h>

#include "fezrestia/ArrayXY.h"
#include "fezrestia/FlagIntBit.h"
#include "fezrestia/BinaryLoader.h"
#include "fezrestia/ImageDds.h"
#include "fezrestia/FileBuffer.h"

using namespace std;
using namespace fezrestia;

namespace GameLib {



enum Cell {
    CELL_WALL,
    CELL_SPACE,
    CELL_GOAL,
    CELL_PLAYER,
    CELL_PLAYER_ON_GOAL,
    CELL_OBJECT,
    CELL_OBJECT_ON_GOAL,
    CELL_UNKNOWN,
};

char cell2Char(Cell cell) {
    switch (cell) {
        case CELL_WALL:
            return '#';
        case CELL_SPACE:
            return ' ';
        case CELL_GOAL:
            return '.';
        case CELL_PLAYER:
            return 'p';
        case CELL_PLAYER_ON_GOAL:
            return 'P';
        case CELL_OBJECT:
            return 'o';
        case CELL_OBJECT_ON_GOAL:
            return 'O';
        case CELL_UNKNOWN:
            return '?';
        default:
            return '!';
    }
}

Cell char2Cell(char c) {
    switch (c) {
        case '#':
            return CELL_WALL;
        case ' ':
            return CELL_SPACE;
        case '.':
            return CELL_GOAL;
        case 'p':
            return CELL_PLAYER;
        case 'P':
            return CELL_PLAYER_ON_GOAL;
        case 'o':
            return CELL_OBJECT;
        case 'O':
            return CELL_OBJECT_ON_GOAL;
        default:
            return CELL_UNKNOWN;
    }
}

const char STAGE_HEADER[] = "RESOLVE STAGE";
const char STAGE_FOOTER[] = "a:LEFT, w:TOP, d:RIGHT, s:BOTTOM, x:EXIT";

const int STAGE_WIDTH = 8;
const int STAGE_HEIGHT = 6;
const int STAGE_SIZE = STAGE_WIDTH * STAGE_HEIGHT;

const int CELL_SIZE = 32;

const int ANIMATION_PROGRESS_STRIDE = 10;

class CellInfo {
public:
    /**
     * CONSTRUCTOR.
     */
    CellInfo() :
            mCell(Cell::CELL_UNKNOWN),
            mPrevDiffX(0),
            mPrevDiffY(0),
            mAnimationProgress(100) {
        // NOP.
    }

    /**
     * DESTRUCTOR.
     */
    ~CellInfo() {
        // NOP.
    }

    /**
     * Set cell.
     *
     * @param cell
     */
    void setCell(Cell cell) {
        mCell = cell;
    }

    /**
     * Get cell
     *
     * @return
     */
    Cell& getCell() {
        return mCell;
    }

    /**
     * Set previous position difference.
     *
     * @param diffX
     * @param diffY
     */
    void setPrevPosDiff(int diffX, int diffY) {
//        cout << "setPrevPosDiff X/Y = " << diffX << '/' << diffY << endl;

        mPrevDiffX = diffX;
        mPrevDiffY = diffY;

        // Reset.
        mAnimationProgress = 0;
    }

    /**
     * Get medium position X, during animation.
     *
     * @param diffPixX [OUT]
     * @param diffPixY [OUT]
     */
    void getAnimatingDiffPixel(
            int* diffPixX,
            int* diffPixY) {
        int maxDiffPixX = mPrevDiffX * CELL_SIZE;
        int maxDiffPixY = mPrevDiffY * CELL_SIZE;

//        cout << "mPrevDiffX/Y = " << mPrevDiffX << '/' << mPrevDiffY << endl;
//        cout << "src diffPixX/Y = " << *diffPixX << '/' << *diffPixY << endl;
//        cout << "mAnimationProgress = " << mAnimationProgress << endl;

        *diffPixX = (100 - mAnimationProgress) * maxDiffPixX / 100;
        *diffPixY = (100 - mAnimationProgress) * maxDiffPixY / 100;

//        cout << "diffPixX/Y = " << *diffPixX << '/' << *diffPixY << endl;

        mAnimationProgress += ANIMATION_PROGRESS_STRIDE;
    }

    /**
     * Now on animating or not.
     *
     * @return
     */
    bool isAnimating() {
        return mAnimationProgress < 100;
    }

private:
    // Cell.
    Cell mCell;
    // Previous position difference.
    int mPrevDiffX;
    int mPrevDiffY;
    // Animation progress from last to current. (0-100)
    int mAnimationProgress;
};

fezrestia::ArrayXY<CellInfo>* gCurrentStage = NULL;
char gInputCode = '\n';
bool gIsUserInputAvailableOnLastFrame = false;

// Stage graphics.
ImageDds* gStageImgDds = NULL;

// FPS
unsigned int gPreviousTimestamp = 0;
const unsigned int FRAME_INTERVAL_MILLIS = 16;

int getPlayerIndex(fezrestia::ArrayXY<CellInfo>* stage) {
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            if ((*stage)(x, y).getCell() == CELL_PLAYER
                    || (*stage)(x, y).getCell() == CELL_PLAYER_ON_GOAL) {
                return y * STAGE_WIDTH + x;
            }
        }
    }

    // Error.
    return -1;
}

int getLeftIndex(int currentIndex) {
    return currentIndex - 1;
}

int getTopIndex(int currentIndex) {
    return currentIndex - STAGE_WIDTH;
}

int getRightIndex(int currentIndex) {
    return currentIndex + 1;
}

int getBottomIndex(int currentIndex) {
    return currentIndex + STAGE_WIDTH;
}

Cell resetCell(Cell cell) {
    switch (cell) {
        case CELL_OBJECT:
            // Fall through.
        case CELL_PLAYER:
            return CELL_SPACE;
        case CELL_OBJECT_ON_GOAL:
            // Fall through.
        case CELL_PLAYER_ON_GOAL:
            return CELL_GOAL;
        default:
            return cell;
    }
}

CellInfo& getCellInfo(fezrestia::ArrayXY<CellInfo>* stage, int index1Axis) {
    return (*stage)(index1Axis % STAGE_WIDTH, index1Axis / STAGE_WIDTH);
}

/**
 * Get XY coordinate from index.
 *
 * @param index [IN]
 * @param x [OUT]
 * @param y [OUT]
 */
void getXY(int index, int* x, int* y) {
    *x = index % STAGE_WIDTH;
    *y = index / STAGE_WIDTH;
}

void updateStage(
        fezrestia::ArrayXY<CellInfo>* stage,
        int playerIndex,
        int step1Index,
        int step2Index) {
    int playerX = 0;
    int playerY = 0;
    int step1X = 0;
    int step1Y = 0;
    int step2X = 0;
    int step2Y = 0;

    getXY(playerIndex, &playerX, &playerY);
    getXY(step1Index, &step1X, &step1Y);
    getXY(step2Index, &step2X, &step2Y);

    switch (getCellInfo(stage, step1Index).getCell()) {
        case CELL_WALL: {
            // NOP. Wall.
            break;
        }
        case CELL_SPACE: {
            // Update dst.
            getCellInfo(stage, step1Index).setCell(Cell::CELL_PLAYER);
            getCellInfo(stage, step1Index).setPrevPosDiff(playerX - step1X, playerY - step1Y);

            // Reset src.
            getCellInfo(stage, playerIndex).setCell(
                    resetCell(getCellInfo(stage, playerIndex).getCell()));
            break;
        }
        case CELL_GOAL: {
            // Update dst.
            getCellInfo(stage, step1Index).setCell(Cell::CELL_PLAYER_ON_GOAL);
            getCellInfo(stage, step1Index).setPrevPosDiff(playerX - step1X, playerY - step1Y);

            // Reset src.
            getCellInfo(stage, playerIndex).setCell(
                    resetCell(getCellInfo(stage, playerIndex).getCell()));
            break;
        }
        case CELL_OBJECT: {
            switch (getCellInfo(stage, step2Index).getCell()) {
                case CELL_WALL:
                    // NOP. Wall.
                    break;
                case CELL_SPACE:
                    // Update dst.
                    getCellInfo(stage, step2Index).setCell(Cell::CELL_OBJECT);
                    getCellInfo(stage, step2Index).setPrevPosDiff(
                            step1X - step2X,
                            step1Y - step2Y);
                    getCellInfo(stage, step1Index).setCell(Cell::CELL_PLAYER);
                    getCellInfo(stage, step1Index).setPrevPosDiff(
                            playerX - step1X,
                            playerY - step1Y);

                    // Reset src.
                    getCellInfo(stage, playerIndex).setCell(
                            resetCell(getCellInfo(stage, playerIndex).getCell()));
                    break;
                case CELL_GOAL:
                    // Update dst
                    getCellInfo(stage, step2Index).setCell(Cell::CELL_OBJECT_ON_GOAL);
                    getCellInfo(stage, step2Index).setPrevPosDiff(
                            step1X - step2X,
                            step1Y - step2Y);
                    getCellInfo(stage, step1Index).setCell(Cell::CELL_PLAYER);
                    getCellInfo(stage, step1Index).setPrevPosDiff(
                            playerX - step1X,
                            playerY - step1Y);

                    // Reset src.
                    getCellInfo(stage, playerIndex).setCell(
                            resetCell(getCellInfo(stage, playerIndex).getCell()));
                    break;
                case CELL_OBJECT:
                    // Fall through.
                case CELL_OBJECT_ON_GOAL:
                    // NOP. 2 objects.
                    break;
                default:
                    // NOP.
                    break;
            }
            break;
        }
        case CELL_OBJECT_ON_GOAL: {
            switch (getCellInfo(stage, step2Index).getCell()) {
                case CELL_WALL:
                    // NOP. Wall.
                    break;
                case CELL_SPACE:
                    // Update dst.
                    getCellInfo(stage, step2Index).setCell(Cell::CELL_OBJECT);
                    getCellInfo(stage, step2Index).setPrevPosDiff(
                            step1X - step2X,
                            step1Y - step2Y);
                    getCellInfo(stage, step1Index).setCell(Cell::CELL_PLAYER_ON_GOAL);
                    getCellInfo(stage, step1Index).setPrevPosDiff(
                            playerX - step1X,
                            playerY - step1Y);

                    // Reset src.
                    getCellInfo(stage, playerIndex).setCell(
                            resetCell(getCellInfo(stage, playerIndex).getCell()));
                    break;
                case CELL_GOAL:
                    // Update dst
                    getCellInfo(stage, step2Index).setCell(Cell::CELL_OBJECT_ON_GOAL);
                    getCellInfo(stage, step2Index).setPrevPosDiff(
                            step1X - step2X,
                            step1Y - step2Y);
                    getCellInfo(stage, step1Index).setCell(Cell::CELL_PLAYER_ON_GOAL);
                    getCellInfo(stage, step1Index).setPrevPosDiff(
                            playerX - step1X,
                            playerY - step1Y);

                    // Reset src.
                    getCellInfo(stage, playerIndex).setCell(
                            resetCell(getCellInfo(stage, playerIndex).getCell()));
                    break;
                case CELL_OBJECT:
                    // Fall through.
                case CELL_OBJECT_ON_GOAL:
                    // NOP. 2 objects.
                    break;
                default:
                    // NOP.
                    break;
            }
            break;
        }
        default: {
            // NOP.
            break;
        }
    }
}

bool isStageClear(fezrestia::ArrayXY<CellInfo>* stage) {
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            if ((*stage)(x, y).getCell() == CELL_OBJECT) {
                // Not clear;
                return false;
            }
        }
    }

    // Cleared.
    return true;
}

void debugCodes() {
    fezrestia::FileBuffer* fb = new fezrestia::FileBuffer("stage.txt");
    cout << "SIZE=" << fb->getSize() << endl;
    for (int i = 0; i < fb->getSize(); ++i) {
        cout << fb->getBuffer()[i];
    }
    fb->release();
    delete fb;
    fb = NULL;

    cout << endl;
    fezrestia::FlagIntBit flag;
    unsigned int flag1 = 1;
    unsigned int flag8 = (1 << 7);
    cout << "Flag1 = " << (flag.isValid(flag1) ? "TRUE" : "FALSE") << endl;
    cout << "Flag8 = " << (flag.isValid(flag8) ? "TRUE" : "FALSE") << endl;
    flag.setValid(flag1);
    flag.setValid(flag8);
    cout << "Flag1 = " << (flag.isValid(flag1) ? "TRUE" : "FALSE") << endl;
    cout << "Flag8 = " << (flag.isValid(flag8) ? "TRUE" : "FALSE") << endl;
    flag.setInvalid(flag1);
    cout << "Flag1 = " << (flag.isValid(flag1) ? "TRUE" : "FALSE") << endl;
    cout << "Flag8 = " << (flag.isValid(flag8) ? "TRUE" : "FALSE") << endl;
    flag.clearAll();
    cout << "Flag1 = " << (flag.isValid(flag1) ? "TRUE" : "FALSE") << endl;
    cout << "Flag8 = " << (flag.isValid(flag8) ? "TRUE" : "FALSE") << endl;
}

void initialize() {
    // Stage.
    fezrestia::FileBuffer* fileBuf = new fezrestia::FileBuffer("stage.txt");
    gCurrentStage = new fezrestia::ArrayXY<CellInfo>(STAGE_WIDTH, STAGE_HEIGHT);
    const char* curBuf = fileBuf->getBuffer();
    int x = 0;
    int y = 0;
    while (y < STAGE_HEIGHT) {
        if (*curBuf == '\n') {
            // Line feed.

            ++y;
            x = 0;
        } else {
            // Some charactor exists.

            if (x < STAGE_WIDTH) {
                // Read stage.
                (*gCurrentStage)(x, y).setCell(char2Cell(*curBuf));
                ++x;
            }
        }

        // Next byte.
        ++curBuf;
    }

    // Release.
    fileBuf->release();
    delete fileBuf;
    fileBuf = NULL;

    if (NULL == gStageImgDds) {
        // Alloc.
        gStageImgDds = new ImageDds("graphics2d.dds");
//        gStageImgDds = new ImageDds("kamo.dds");
    }

    // Timestamp.
    gPreviousTimestamp = 0;
}

void getInput() {
    // Reset.
    gInputCode = '\n';

    // Check moving.
    if (getCellInfo(gCurrentStage, getPlayerIndex(gCurrentStage)).isAnimating()) {
        // NOP. Now on animating.
        return;
    }

    if (Framework::instance().isKeyOn('\n')) {
        gInputCode = '\n';
        gIsUserInputAvailableOnLastFrame = false;
    } else if (Framework::instance().isKeyOn('a')) {
        if (!gIsUserInputAvailableOnLastFrame) {
            gInputCode = 'a';
        }
        gIsUserInputAvailableOnLastFrame = true;
    } else if (Framework::instance().isKeyOn('w')) {
        if (!gIsUserInputAvailableOnLastFrame) {
            gInputCode = 'w';
        }
        gIsUserInputAvailableOnLastFrame = true;
    } else if (Framework::instance().isKeyOn('d')) {
        if (!gIsUserInputAvailableOnLastFrame) {
            gInputCode = 'd';
        }
        gIsUserInputAvailableOnLastFrame = true;
    } else if (Framework::instance().isKeyOn('s')) {
        if (!gIsUserInputAvailableOnLastFrame) {
            gInputCode = 's';
        }
        gIsUserInputAvailableOnLastFrame = true;
    } else if (Framework::instance().isKeyOn('x')) {
        if (!gIsUserInputAvailableOnLastFrame) {
            gInputCode = 'x';
        }
        gIsUserInputAvailableOnLastFrame = true;
    } else {
        // Not supported.
        gInputCode = '\n';
        gIsUserInputAvailableOnLastFrame = false;
    }
}

void updateState() {
    // Position.
    int playerIndex = -1;
    int step1Index = -1;
    int step2Index = -1;

    // Input.
    switch (gInputCode) { // Main loop.
        case '\n': {
            // NOP. Nothing input.
            break;
        }
        case 'a': {
            // Left.
            playerIndex = getPlayerIndex(gCurrentStage);
            step1Index = getLeftIndex(playerIndex);
            step2Index = getLeftIndex(step1Index);
            break;
        }
        case 'w': {
            // Top.
            playerIndex = getPlayerIndex(gCurrentStage);
            step1Index = getTopIndex(playerIndex);
            step2Index = getTopIndex(step1Index);
            break;
        }
        case 'd': {
            // Right.
            playerIndex = getPlayerIndex(gCurrentStage);
            step1Index = getRightIndex(playerIndex);
            step2Index = getRightIndex(step1Index);
            break;
        }
        case 's': {
            // Bottom.
            playerIndex = getPlayerIndex(gCurrentStage);
            step1Index = getBottomIndex(playerIndex);
            step2Index = getBottomIndex(step1Index);
            break;
        }
        case 'x': {
            // Finish.
//            cout << "EXIT" << endl;
            Framework::instance().requestEnd();
            break;
        }
        default: {
            // NOP. Unsupported.
            cout << "Unsupported Operation." << endl;
            break;
        }
    }

    // Update.
    updateStage(
            gCurrentStage,
            playerIndex,
            step1Index,
            step2Index);
}

void renderConsole() {
    // Clear.
//    system("cls");

    // Render stage.
    cout << STAGE_HEADER << endl;
    cout << endl;
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            cout << cell2Char((*gCurrentStage)(x, y).getCell());
        }
        cout << endl;
    }
    cout << endl;
    cout << STAGE_FOOTER << endl;
}

void renderScreenCell(int x, int y, unsigned int color) {
    unsigned int* vram = Framework::instance().videoMemory();
    int screenWidth = Framework::instance().width();

    const int fixedSize = 16;

    for (int i = 0; i < fixedSize; ++i) {
        for (int j = 0; j < fixedSize; ++j) {
            vram[(y * fixedSize + i) * screenWidth + (x * fixedSize + j)] = color;
        }
    }
}

void renderScreenCellWithGraphics(int x, int y, Cell cell, int diffX, int diffY) {
    unsigned int* vram = Framework::instance().videoMemory();
    int screenWidth = Framework::instance().width();
    int screenHeight = Framework::instance().height();

    unsigned int srcX = CELL_SIZE * cell; // Use enum ordinal.
    unsigned int srcY = 0;

    unsigned int dstX = x * CELL_SIZE + diffX;
    unsigned int dstY = y * CELL_SIZE + diffY;

    gStageImgDds->render(
            srcX,
            srcY,
            srcX + CELL_SIZE,
            srcY + CELL_SIZE,
            vram,
            screenWidth,
            screenHeight,
            dstX,
            dstY);
}

void renderScreenCellWithGraphics(int x, int y, Cell cell) {
    renderScreenCellWithGraphics(x, y, cell, 0, 0);
}

void renderScreen() {
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            Cell cell = (*gCurrentStage)(x, y).getCell();
            unsigned int color = 0;

            // Get color.
            switch (cell) {
                case CELL_WALL:
                    color = 0xFFFFFF;
                    break;

                case CELL_SPACE:
                    color = 0x000000;
                    break;

                case CELL_GOAL:
                    color = 0x0000FF;
                    break;

                case CELL_PLAYER: 
                    color = 0x00FF00;
                    break;

                case CELL_PLAYER_ON_GOAL:
                    color = 0x00FFFF;
                    break;

                case CELL_OBJECT:
                    color = 0xFF0000;
                    break;

                case CELL_OBJECT_ON_GOAL:
                    color = 0xFF00FF;
                    break;

                case CELL_UNKNOWN:
                    color = 0xAAAAAA;
                    break;

                default:
                    // Unexpected.
                    color = 0xAAAAAA;
                    break;
            }

            // Render.
            renderScreenCell(x, y, color);
        }
    }
}

void renderScreenWithGraphics() {
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            CellInfo& cellInfo = (*gCurrentStage)(x, y);

            // Base rendering.
            switch (cellInfo.getCell()) {
                case CELL_WALL:
                    // Fall through.
                case CELL_GOAL:
                    renderScreenCellWithGraphics(x, y, CELL_SPACE);
                    renderScreenCellWithGraphics(x, y, cellInfo.getCell());
                    break;

                case CELL_PLAYER: 
                    // Fall through.
                case CELL_OBJECT:
                    // Base rendering only.
                    renderScreenCellWithGraphics(x, y, CELL_SPACE);
                    break;

                case CELL_PLAYER_ON_GOAL:
                    renderScreenCellWithGraphics(x, y, CELL_SPACE);
                    renderScreenCellWithGraphics(x, y, CELL_GOAL);
                    break;

                case CELL_OBJECT_ON_GOAL:
                    renderScreenCellWithGraphics(x, y, CELL_SPACE);
                    renderScreenCellWithGraphics(x, y, CELL_GOAL);
                    break;

                default:
                    renderScreenCellWithGraphics(x, y, cellInfo.getCell());
                    break;
            }
        }
    }

    // Rendering player and object.
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            CellInfo& cellInfo = (*gCurrentStage)(x, y);

            switch (cellInfo.getCell()) {
                case CELL_PLAYER:
                    // Fall through.
                case CELL_OBJECT:
                    // Fall through.
                case CELL_PLAYER_ON_GOAL:
                    // Fall through.
                case CELL_OBJECT_ON_GOAL: {
                    int diffX = 0;
                    int diffY = 0;
                    if (cellInfo.isAnimating()) {
                        cellInfo.getAnimatingDiffPixel(&diffX, &diffY);
                    }

                    Cell renderingCell = cellInfo.getCell();
                    switch (renderingCell) {
                        case CELL_PLAYER_ON_GOAL:
                            renderingCell = Cell::CELL_PLAYER;
                            break;
                        case CELL_OBJECT_ON_GOAL:
                            renderingCell = Cell::CELL_OBJECT;
                            break;
                        default:
                            // NOP.
                            break;
                    }

                    renderScreenCellWithGraphics(
                            x,
                            y,
                            renderingCell,
                            diffX,
                            diffY);
                    break;
                }

                default:
                    // NOP.
                    break;
            }
        }
    }
}

void render() {
//    renderConsole();
//    renderScreen();
    renderScreenWithGraphics();
}

void finalize() {
    // Release.
    delete gCurrentStage;
    gCurrentStage = NULL;

    delete gStageImgDds;
    gStageImgDds = NULL;
}



void Framework::update() {
    // Initializer.
    if (gCurrentStage == NULL) {
        initialize();
        render();
        return;
    }

    // FPS.
    unsigned int currentTimestamp = Framework::instance().time();
    if (gPreviousTimestamp != 0) {
        unsigned int interval = currentTimestamp - gPreviousTimestamp;
        unsigned int fps = 1000 / interval;
        cout << "FPS=" << fps << endl;
    }
    gPreviousTimestamp = currentTimestamp;

    // Check clear.
    bool isCleared = isStageClear(gCurrentStage);
    if (isCleared) {
        cout << endl << "CONGRATURATIONS !" << endl;
    } else {
        // NOP. Wait for next input.
    }

    getInput();
    updateState();
    render();

    // FPS.
    unsigned int outTimestamp = Framework::instance().time();
    int elapsedTime = outTimestamp - currentTimestamp;
    int sleepTime = FRAME_INTERVAL_MILLIS - elapsedTime;
    if (0 < sleepTime) {
        cout << "Sleep=" << sleepTime << endl;
        Framework::sleep(sleepTime);
    }

    // Finalizer.
    if (Framework::instance().isEndRequested()) {
        finalize();
    }
}



} // namespace GameLib
