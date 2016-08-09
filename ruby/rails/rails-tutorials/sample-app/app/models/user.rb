class User < ActiveRecord::Base

  has_secure_password

  attr_accessor :remember_token
  attr_accessor :activation_token
  attr_accessor :reset_token


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

  # Before save callback.
  #
  before_save :downcase_email

  # Before create callback.
  #
  before_create :create_activation_digest

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
  def authenticated?(attribute, token)
    digest = send("#{attribute}_digest")
    return false if digest.nil?
    return BCrypt::Password.new(digest).is_password?(token)
  end

  # Remove user's log in token
  #
  def forget
    update_attribute(:remember_digest, nil)
  end

  # Activate user.
  #
  def activate
    update_columns(activated: true, activated_at: Time.zone.now)
  end

  # Send activation mail.
  #
  def send_activation_email
    UserMailer.account_activation(self).deliver_now
  end

  # Create password reset digest.
  #
  def create_reset_digest
    self.reset_token = User.new_token
    update_columns(
        reset_digest: User.digest(reset_token),
        reset_sent_at: Time.zone.now)
  end

  # Send password reset mail.
  #
  def send_password_reset_email
    UserMailer.password_reset(self).deliver_now
  end

  # Check password reset is valid or not.
  #
  def password_reset_expired?
    reset_sent_at < 2.hours.ago
  end

  private

    def downcase_email
      self.email = email.downcase
    end

    def create_activation_digest
      self.activation_token = User.new_token
      self.activation_digest = User.digest(activation_token)
    end

end

