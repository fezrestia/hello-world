class SampleController < ApplicationController

  def index
    # NOP
  end

  def new
    # NOP
  end

  def create
    render plain: "PARAS: key1=#{params[:data][:key1]}, key2=#{params[:data][:key2]}"
  end

end
