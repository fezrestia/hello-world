import * as React from "react";
import * as ReactDOM from "react-dom";

import { Game } from "./components/Game";
import { Clock } from "./components/Clock";
import { LoginForm } from "./components/LoginForm";

ReactDOM.render(<Clock />, document.getElementById("clock"));

ReactDOM.render(<LoginForm />, document.getElementById("login"));

ReactDOM.render(
  <Game />,
  document.getElementById('root')
);
