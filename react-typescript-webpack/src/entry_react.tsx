import * as React from "react";
import * as ReactDOM from "react-dom";

import { Game } from "./components/Game";
import { Clock } from "./components/Clock";

ReactDOM.render(<Clock />, document.getElementById("clock"));

ReactDOM.render(
  <Game />,
  document.getElementById('root')
);
