class User < ActiveRecord::Base

  has_secure_password

  attr_accessor :remember_token

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
      length: { minimum: 6 },
      allow_nil: true

  # Callback
  #
  before_save {
#    self.email = email.downcase
    email.downcase!
  }

  # Return hashed string.
  #
  def User.digest(string)
    cost = ActiveModel::SecurePassword.min_cost ?
        BCrypt::Engine::MIN_COST : BCrypt::Engine.cost
    return BCrypt::Password.create(string, cost: cost)
  end

  # Return random token string.
  #
  def User.new_token
    return SecureRandom.urlsafe_base64
  end

  # Remember log in session as token.
  #
  def remember
    self.remember_token = User.new_token
    update_attribute(:remember_digest, User.digest(remember_token))
  end

  # Check remember token is valid or not.
  #
  def authenticated?(remember_token)
    return false if remember_digest.nil?
    return BCrypt::Password.new(remember_digest).is_password?(remember_token)
  end

  # Remove user's log in token
  #
  def forget
    update_attribute(:remember_digest, nil)
  end

end

