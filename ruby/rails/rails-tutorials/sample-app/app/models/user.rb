class User < ActiveRecord::Base

  has_secure_password

  # Validation
  #
  VALID_EMAIL_REGEX = /\A[\w+\-.]+@[a-z\d\-]+(\.[a-z\d\-]+)*\.[a-z]+\z/i
  validates :name,
      presence: true,
      length: { maximum: 50 }
  validates :email,
      presence: true,
      length: { maximum: 255 },
      format: { with: VALID_EMAIL_REGEX },
      uniqueness: { case_sensitive: false }
  validates :password,
      presence: true,
      length: { minimum: 6 }

  # Callback
  #
  before_save {
#    self.email = email.downcase
    email.downcase!
  }

end

