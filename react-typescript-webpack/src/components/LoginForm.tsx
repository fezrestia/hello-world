import * as React from "react";

interface Props {
}

interface State {
  user: string,
  pass: string,
  text: string,
  select: string,
}

enum Options {
  HOGE = "hoge",
  FUGA = "fuga",
  PIYO = "piyo",
}

export class LoginForm extends React.Component<Props, State> {

  static readonly ID_USER_INPUT = "user_input";
  static readonly ID_PASS_INPUT = "pass_input";
  static readonly ID_TEXT_INPUT = "text_input";

  constructor(props: Props) {
    super(props);

    this.state = {
      user: "",
      pass: "",
      text: "default",
      select: Options.HOGE,
    };

    this.handleChange = this.handleChange.bind(this);
    this.handleTextChange = this.handleTextChange.bind(this);
    this.handleSelectChange = this.handleSelectChange.bind(this);
    this.handleSubmit = this.handleSubmit.bind(this);

  }

  handleChange(event: React.ChangeEvent<HTMLInputElement>) {
    console.log("handleChange()")

    let targetId = event.target.id;
    let value = event.target.value;

    switch (targetId) {
      case LoginForm.ID_USER_INPUT:
        this.setState( { user: value } );
        break;

      case LoginForm.ID_PASS_INPUT:
        this.setState( { pass: value } );
        break;
    }
  }

  handleTextChange(event: React.ChangeEvent<HTMLTextAreaElement>) {
    console.log("handleTextChange()");

    let targetId = event.target.id;
    let value = event.target.value;

    switch (targetId) {
      case LoginForm.ID_TEXT_INPUT:
        this.setState( { text: value } );
        break;
    }
  }

  handleSelectChange(event: React.ChangeEvent<HTMLSelectElement>) {
    console.log("handleSelectChange()");

    let value = event.target.value;
    this.setState( { select: value } );
  }

  handleSubmit(event: React.FormEvent<HTMLFormElement>) {
    console.log("handleSubmit()")

    alert(`User/Pass is submitted : ${this.state.user}/${this.state.pass}`)

    event.preventDefault();
  }

  render() {
    return (
      <form onSubmit={this.handleSubmit} >
        <label>
          USER: <input id={LoginForm.ID_USER_INPUT} type="text" value={this.state.user} onChange={this.handleChange} />
        </label>
        <br />
        <label>
          PASS: <input id={LoginForm.ID_PASS_INPUT} type="text" value={this.state.pass} onChange={this.handleChange} />
        </label>
        <br />
        <textarea id={LoginForm.ID_TEXT_INPUT} value={this.state.text} onChange={this.handleTextChange} />
        <br />
        <select value={this.state.select} onChange={this.handleSelectChange} >
          {
            Object.entries(Options).map( ([key, val]) => {
                return ( <option key={key} value={key} >{key}</option> );
            } )
          }
        </select>
        <br />
        <input type="submit" value="Login" />
      </form>
    );
  }

}
