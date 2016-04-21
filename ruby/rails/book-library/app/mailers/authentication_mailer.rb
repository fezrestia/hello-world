class AuthenticationMailer < ApplicationMailer

    default from: "system@b.c"

    def complete_registration(user)
        @user = user
        mail {
            to: user.email,
            subject: 'User registration is completed.'
        }
    end

end

