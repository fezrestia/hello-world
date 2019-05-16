import * as React from "react";

interface Props {
  value: string;
  onClick: () => void,
}

export function Square(props: Props) {
  return (
    <button
        className="square"
        onClick={ props.onClick }
    >
      { props.value }
    </button>
  );
}

/**
interface Props {
  value: string;
  onClick: () => void,
}

interface State {
}

export class Square extends React.Component<Props, State> {
  render() {
    return (
      <button
          className="square"
          onClick={this.props.onClick}
      >
        {this.props.value}
      </button>
    );
  }
}
*/
