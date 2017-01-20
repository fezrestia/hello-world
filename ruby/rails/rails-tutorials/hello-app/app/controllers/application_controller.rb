class ApplicationController < ActionController::Base
  # Prevent CSRF attacks by raising an exception.
  # For APIs, you may want to use :null_session instead.
  protect_from_forgery with: :exception

  def hello
    render text: "HELLO WORLD !"
  end

  def goodbye
    render text: "Goodbye World !"
  end

  def hola
    render text: "¡Hola, mundo!"
  end

end
