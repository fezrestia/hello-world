import * as React from "react";

import { Square } from "./Square";

interface Props {
  curPlayer: string,
  squares: Array<string>,
  handleClick: (index: number) => void,
}

interface State {
}

export class Board extends React.Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = {
    };
  }

  renderSquare(i: number) {
    return (
      <Square
          value={this.props.squares[i]}
          onClick={ () => { this.props.handleClick(i) } }
      />
    );
  }

  render() {
    return (
      <div>
        <div className="board-row" >
          {this.renderSquare(0)}
          {this.renderSquare(1)}
          {this.renderSquare(2)}
        </div>
        <div className="board-row" >
          {this.renderSquare(3)}
          {this.renderSquare(4)}
          {this.renderSquare(5)}
        </div>
        <div className="board-row" >
          {this.renderSquare(6)}
          {this.renderSquare(7)}
          {this.renderSquare(8)}
        </div>

      </div>
    );
  }
}
