import * as React from "react";
import { ClockLabelContext } from "../context";

interface Props {
}

interface State {
  date: Date,
}

export class Clock extends React.Component<Props, State> {

  timerId: number = 0;

  static contextType = ClockLabelContext;

  constructor(props: Props) {
    super(props);

    this.state = {
        date: new Date(),
    };

  }

  componentDidMount() {
    this.timerId = window.setInterval(
        () => { this.tick() },
        1000);
  }

  componentWillUnmount() {
    window.clearInterval(this.timerId);
  }

  tick() {
    this.setState( (state, props) => {
      return (
          {
              date: new Date(),
          }
      );
    });
  }

  render() {
    return (
        <div>
          <p>{this.context} : {this.state.date.toLocaleTimeString()}</p>
        </div>
    );
  }
}
