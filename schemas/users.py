import re
from pydantic import BaseModel, Field, field_validator, model_validator

_MOBILE_RE = re.compile(r"^1[3-9]\d{9}$")


def _validate_password_mixed(value: str) -> str:
    if not (re.search(r"[A-Za-z]", value) and re.search(r"\d", value)):
        raise ValueError("密码必须同时包含字母和数字")
    return value


def _strip_required(value, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name}不能为空")
    result = value.strip()
    if not result:
        raise ValueError(f"{field_name}不能为空")
    return result


def _required_no_strip(value, field_name: str) -> str:
    if not isinstance(value, str) or value == "":
        raise ValueError(f"{field_name}不能为空")
    return value


class UserRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="用户名")
    telNum: str = Field(..., min_length=11, max_length=11, description="电话号码")
    password: str = Field(..., min_length=6, max_length=128, description="密码")

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value) -> str:
        return _strip_required(value, "用户名")

    @field_validator("telNum", mode="before")
    @classmethod
    def validate_tel_num(cls, value) -> str:
        value = _strip_required(value, "电话号码")
        if not _MOBILE_RE.fullmatch(value):
            raise ValueError("手机号格式不正确")
        return value

    @field_validator("password", mode="before")
    @classmethod
    def validate_password(cls, value) -> str:
        value = _required_no_strip(value, "密码")
        return _validate_password_mixed(value)


class UserLoginRequest(BaseModel):
    telNum: str = Field(..., min_length=11, max_length=11, description="电话号码")
    password: str = Field(..., min_length=1, max_length=128, description="密码")

    @field_validator("telNum", mode="before")
    @classmethod
    def validate_tel_num(cls, value) -> str:
        value = _strip_required(value, "电话号码")
        if not _MOBILE_RE.fullmatch(value):
            raise ValueError("手机号格式不正确")
        return value

    @field_validator("password", mode="before")
    @classmethod
    def validate_password(cls, value) -> str:
        return _required_no_strip(value, "密码")


class UserUpdateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="用户名")
    telNum: str = Field(..., min_length=11, max_length=11, description="电话号码")

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, value) -> str:
        return _strip_required(value, "用户名")

    @field_validator("telNum", mode="before")
    @classmethod
    def validate_tel_num(cls, value) -> str:
        value = _strip_required(value, "电话号码")
        if not _MOBILE_RE.fullmatch(value):
            raise ValueError("手机号格式不正确")
        return value


class UserUpdatePassword(BaseModel):
    oldPassword: str = Field(..., min_length=1, max_length=128, description="旧密码")
    password: str = Field(..., min_length=6, max_length=128, description="新密码")

    @field_validator("oldPassword", mode="before")
    @classmethod
    def validate_old_password(cls, value) -> str:
        return _required_no_strip(value, "旧密码")

    @field_validator("password", mode="before")
    @classmethod
    def validate_new_password(cls, value) -> str:
        value = _required_no_strip(value, "新密码")
        return _validate_password_mixed(value)

    @model_validator(mode="after")
    def check_new_password(self):
        if self.oldPassword == self.password:
            raise ValueError("新密码不能与旧密码相同")
        return self
