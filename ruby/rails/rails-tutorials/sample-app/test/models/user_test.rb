require 'test_helper'

class UserTest < ActiveSupport::TestCase

  def setup
    @user = User.new(
        name: "Test User",
        email: "user@test.com",
        password: "hogefugapiyo",
        password_confirmation: "hogefugapiyo")
  end

  test "should be valid" do
    assert @user.valid?
  end

  test "name should be present" do
    @user.name = "    "
    assert_not @user.valid?
  end

  test "email should be present" do
    @user.email = "    "
    assert_not @user.valid?
  end

  test "name should not be too long" do
    @user.name = "a" * 51
    assert_not @user.valid?
  end

  test "email should not too long" do
    @user.email = "a" * 247 + "@test.com"
    assert_not @user.valid?
  end

  test "email validation should accept valid addresses" do
    valid_addresses = %w[
        user@example.com
        USER@foo.COM
        A_US-ER@foo.bar.org
        first.last@foo.jp
        alice+bob@baz.cn
    ]

    valid_addresses.each do |valid_address|
      @user.email = valid_address
      assert @user.valid?, "#{valid_address.inspect} sould be valid"
    end
  end

  test "email validation should reject invalid addresses" do
    invalid_addresses = %w[
        user@example,com
        user_at_foo.org
        user.name@example.
        foo@bar_baz.com
        foo@bar+baz.com
        hoge@fuga..piyo
        hoge@fuga...piyo
    ]

    invalid_addresses.each do |invalid_address|
      @user.email = invalid_address
      assert_not @user.valid? "${invalid_address.inspect} should be invalid"
    end
  end

  test "email addresses should be unique" do
    duplicate_user = @user.dup
    duplicate_user.email = @user.email.upcase
    @user.save
    assert_not duplicate_user.valid?
  end

  test "email address should be saved as lower-case" do
    mixed_case_email = "HOGE@FuGa.pIyO"
    @user.email = mixed_case_email
    @user.save
    assert_equal mixed_case_email.downcase, @user.reload.email
  end

  test "password should be present (nonblank)" do
    @user.password = @user.password_confirmation = " " * 6
    assert_not @user.valid?
  end

  test "password should have a minimum length" do
    @user.password = @user.password_confirmation = "a" * 5
    assert_not @user.valid?
  end

  test "authenticated? should return false for a user with nil digest" do
    assert_not @user.authenticated?(:remember, '')
  end

  test "associated microposts should be destroyed" do
    @user.save
    @user.microposts.create!(content: "Lorem ipsum")
    assert_difference 'Micropost.count', -1 do
      @user.destroy
    end
  end

  test "should follow and unfollow a user" do
    rachael = users(:rachael)
    lana = users(:lana)
    assert_not rachael.following?(lana)
    rachael.follow(lana)
    assert rachael.following?(lana)
    assert lana.followers.include?(rachael)
    rachael.unfollow(lana)
    assert_not rachael.following?(lana)
  end

  test "feed should have the right posts" do
    rachael = users(:rachael)
    kanna = users(:kanna)
    mallory = users(:mallory)

    # Check following feed.
    kanna.microposts.each do |post_following|
      assert rachael.feed.include?(post_following)
    end

    # Check self feed.
    rachael.microposts.each do |post_self|
      assert rachael.feed.include?(post_self)
    end

    # Check un-follow feed.
    mallory.microposts.each do |post_unfollowed|
      assert_not rachael.feed.include?(post_unfollowed)
    end
  end

end

