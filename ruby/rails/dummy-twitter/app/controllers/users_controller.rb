class UsersController < ApplicationController
    def index
    end

    def show

        @user = User.find_by(:username => params[:username])

#        if params[:username] == 'ryooopan'
#            @user = User.find_by(:username => 'ryooopan')
#        elsif params[:username] == 'moyahima'
#            @user = User.find_by(:username => 'moyahima')
#        end

#        @user = Hash.new
#        if params[:username] == 'ryooopen'
#            @user[:name] = 'Ryo Suzuki'
#            @user[:username] = 'ryooopen'
#            @user[:location] = 'Kanaagwa, Japan'
#            @user[:about] = 'Hello'
#        elsif params[:username] == 'moyahima'
#            @user[:name] = 'Shohei Aoki'
#            @user[:username] = 'moyahima'
#            @user[:location] = 'Tottori, Japan'
#            @user[:about] = 'nice'
#        end

    end

end

