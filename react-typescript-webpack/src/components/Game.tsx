import * as React from "react";

import { Board } from "./Board";

interface Props {
}

interface State {
  curPlayer: string,
  histories: Array<string>[],
}

const EMPTY = '';
const MARU = 'O';
const BATU = 'X';

export class Game extends React.Component<Props, State> {

  constructor(props: Props) {
    super(props);

    this.state = {
        curPlayer: this.getPlayer(0),
        histories: [new Array<string>(9).fill(EMPTY)],
    };
  }

  getPlayer(curStep: number): string {
    return curStep % 2 == 0 ? MARU : BATU;
  }

  getCurStep(): number {
    return this.state.histories.length - 1;
  }

  getCurSquares(): Array<string> {
    return this.state.histories[this.state.histories.length - 1];
  }

  render() {
    const curSquares = this.getCurSquares();
    const winner = calcWinner(curSquares);

    let statusLine: string;
    if (winner != null) {
      statusLine = `WINNER: ${winner}`;
    } else {
      statusLine = `NEXT: ${this.state.curPlayer}`;
    }

    const steps = this.state.histories.map( (squares, step) => {
      let desc: string;
      if (step == 0) {
        desc = `Go to START`;
      } else {
        desc = `Go to STEP #${step}`;
      }

      return (
          <li key={step} >
            <button onClick={ () => this.jumpTo(step) } >{desc}</button>
          </li>
      );
    });

    return (
      <div className="game">
        <div className="game-board">
          <Board
              curPlayer={this.state.curPlayer}
              squares={curSquares}
              handleClick={ (index: number) => { this.handleClick(index) } }
          />
        </div>
        <div className="status" >
          {statusLine}
        </div>
        <div className="game-info">
          <ol>{steps}</ol>
        </div>
      </div>
    );
  }

  handleClick(index: number) {
    const curSquares = this.getCurSquares();
    const winner = calcWinner(curSquares);

    if (winner != null || curSquares[index] != EMPTY) {
      return;
    }

    const nextSquares = curSquares.slice();
    nextSquares[index] = this.state.curPlayer;

    const nextStep = this.getCurStep() + 1;
    let nextPlayer = this.getPlayer(nextStep);

    let newHistories = this.state.histories;
    newHistories.push(nextSquares);

    this.setState({
        curPlayer: nextPlayer,
        histories: newHistories,
    });
  }

  jumpTo(step: number) {
    const newPlayer = this.getPlayer(step);
    const newHistories = this.state.histories.slice(0, step + 1);

    this.setState({
        curPlayer: newPlayer,
        histories: newHistories,
    });
  }
}

function calcWinner(sqrs: Array<string>): string|null {
  const lines = [
      [0, 1, 2],
      [3, 4, 5],
      [6, 7, 8],
      [0, 3, 6],
      [1, 4, 7],
      [2, 5, 8],
      [0, 4, 8],
      [2, 4, 6],
  ];

  for (let i = 0; i < lines.length; i++) {
    const [a, b, c] = lines[i];

    if (sqrs[a] != EMPTY && sqrs[a] == sqrs[b] && sqrs[a] == sqrs[c]) {
      return sqrs[a];
    }
  }

  return null;
}
