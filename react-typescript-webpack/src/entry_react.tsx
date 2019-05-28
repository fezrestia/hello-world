import * as React from "react";
import * as ReactDOM from "react-dom";

import { Game } from "./components/Game";
import { Clock } from "./components/Clock";
import { LoginForm } from "./components/LoginForm";

import { ClockLabelContext } from "./context";

ReactDOM.render(<Clock />, document.getElementById("clock"));

let customLabel = "NOW";
let clockElm = (
  <ClockLabelContext.Provider value={customLabel} >
    <Clock />
  </ClockLabelContext.Provider>
);

ReactDOM.render(clockElm, document.getElementById("labelled_clock"));

ReactDOM.render(<LoginForm />, document.getElementById("login"));

ReactDOM.render(
  <Game />,
  document.getElementById('root')
);
