#include <iostream>
#include <stdlib.h>

#include "util.h"
#include "ArrayXY.h"
#include "FlagIntBit.h"

using namespace std;



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

const int STAGE_WIDTH = 8;
const int STAGE_HEIGHT = 6;
const int STAGE_SIZE = STAGE_WIDTH * STAGE_HEIGHT;

//const char BASE_STAGE[] = "\
//########\
//#      #\
//# .. p #\
//# oo   #\
//#      #\
//########\
//";

char STAGE_FOOTER[] = "a:LEFT, w:TOP, d:RIGHT, s:BOTTOM, x:EXIT";



int getPlayerIndex(fezrestia::ArrayXY<Cell>* stage) {
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            if ((*stage)(x, y) == CELL_PLAYER || (*stage)(x, y) == CELL_PLAYER_ON_GOAL) {
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

Cell& getCell(fezrestia::ArrayXY<Cell>* stage, int index1Axis) {
    return (*stage)(index1Axis % STAGE_WIDTH, index1Axis / STAGE_WIDTH);
}

void updateStageState(
        fezrestia::ArrayXY<Cell>* stage,
        int playerIndex,
        int step1Index,
        int step2Index) {
    switch (getCell(stage, step1Index)) {
        case CELL_WALL: {
            // NOP. Wall.
            break;
        }
        case CELL_SPACE: {
            // Update dst.
            getCell(stage, step1Index) = CELL_PLAYER;
            // Reset src.
            getCell(stage, playerIndex) = resetCell(getCell(stage, playerIndex));
            break;
        }
        case CELL_GOAL: {
            // Update dst.
            getCell(stage, step1Index) = CELL_PLAYER_ON_GOAL;
            // Reset src.
            getCell(stage, playerIndex) = resetCell(getCell(stage, playerIndex));
            break;
        }
        case CELL_OBJECT: {
            switch (getCell(stage, step2Index)) {
                case CELL_WALL:
                    // NOP. Wall.
                    break;
                case CELL_SPACE:
                    // Update dst.
                    getCell(stage, step2Index) = CELL_OBJECT;
                    getCell(stage, step1Index) = CELL_PLAYER;
                    // Reset src.
                    getCell(stage, playerIndex) = resetCell(getCell(stage, playerIndex));
                    break;
                case CELL_GOAL:
                    // Update dst
                    getCell(stage, step2Index) = CELL_OBJECT_ON_GOAL;
                    getCell(stage, step1Index) = CELL_PLAYER;
                    // Reset src.
                    getCell(stage, playerIndex) = resetCell(getCell(stage, playerIndex));
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
            switch (getCell(stage, step2Index)) {
                case CELL_WALL:
                    // NOP. Wall.
                    break;
                case CELL_SPACE:
                    // Update dst.
                    getCell(stage, step2Index) = CELL_OBJECT;
                    getCell(stage, step1Index) = CELL_PLAYER_ON_GOAL;
                    // Reset src.
                    getCell(stage, playerIndex) = resetCell(getCell(stage, playerIndex));
                    break;
                case CELL_GOAL:
                    // Update dst
                    getCell(stage, step2Index) = CELL_OBJECT_ON_GOAL;
                    getCell(stage, step1Index) = CELL_PLAYER_ON_GOAL;
                    // Reset src.
                    getCell(stage, playerIndex) = resetCell(getCell(stage, playerIndex));
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

void renderScreen(fezrestia::ArrayXY<Cell>* stage) {
    // Clear.
//    system("cls");

    // Render stage.
    cout << STAGE_HEADER << endl;
    cout << endl;
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            cout << cell2Char((*stage)(x, y));
        }
        cout << endl;
    }
    cout << endl;
    cout << STAGE_FOOTER << endl;
}

bool isStageClear(fezrestia::ArrayXY<Cell>* stage) {
    for (int y = 0; y < STAGE_HEIGHT; ++y) {
        for (int x = 0; x < STAGE_WIDTH; ++x) {
            if ((*stage)(x, y) == CELL_OBJECT) {
                // Not clear;
                return false;
            }
        }
    }

    // Cleared.
    return true;
}

int main() {

/// DEBUG
    fezrestia::FileBuffer* fb = fezrestia::loadFile("stage.txt");
    cout << "SIZE=" << fb->getSize() << endl;
    cout.write(fb->getBuffer(), fb->getSize());
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
/// DEBUG

    // Input char.
    char input = '\n';
    // Main loop active flag.
    bool isActive = true;

    // Stage.
    fezrestia::FileBuffer* fileBuf = fezrestia::loadFile("stage.txt");
    fezrestia::ArrayXY<Cell>* curStage = new fezrestia::ArrayXY<Cell>(STAGE_WIDTH, STAGE_HEIGHT);
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
                (*curStage)(x, y) = char2Cell(*curBuf);
                ++x;
            }
        }

        // Next byte.
        ++curBuf;
    }

    // Initial screen.
    renderScreen(curStage);

    while(isActive) {
        // Wait for user input.
        cin >> input;

        // Position.
        int playerIndex = -1;
        int step1Index = -1;
        int step2Index = -1;

        // Input.
        switch (input) { // Main loop.
            case '\n': {
                // NOP. Do not input anything.
                break;
            }
            case 'a': {
                // Left.
                playerIndex = getPlayerIndex(curStage);
                step1Index = getLeftIndex(playerIndex);
                step2Index = getLeftIndex(step1Index);
                break;
            }
            case 'w': {
                // Top.
                playerIndex = getPlayerIndex(curStage);
                step1Index = getTopIndex(playerIndex);
                step2Index = getTopIndex(step1Index);
                break;
            }
            case 'd': {
                // Right.
                playerIndex = getPlayerIndex(curStage);
                step1Index = getRightIndex(playerIndex);
                step2Index = getRightIndex(step1Index);
                break;
            }
            case 's': {
                // Bottom.
                playerIndex = getPlayerIndex(curStage);
                step1Index = getBottomIndex(playerIndex);
                step2Index = getBottomIndex(step1Index);
                break;
            }
            case 'x': {
                // Finish.
                isActive = false;
                cout << "EXIT" << endl;
                break;
            }
            default: {
                // NOP. Unsupported.
                cout << "Unsupported Operation." << endl;
                break;
            }
        }

        // Update.
        updateStageState(
                curStage,
                playerIndex,
                step1Index,
                step2Index);

        // Render.
        renderScreen(curStage);

        // Check clear.
        bool isCleared = isStageClear(curStage);
        if (isCleared) {
            isActive = false;
            cout << endl << "CONGRATURATIONS !" << endl;
        } else {
            // NOP. Wait for next input.
        }
    } // Main loop.

    // Release.
    delete curStage;
    curStage = NULL;
    fileBuf->release();
    delete fileBuf;
    fileBuf = NULL;

    // Wait.
    while (true) {
        // NOP.
    }

    return 0;
}
