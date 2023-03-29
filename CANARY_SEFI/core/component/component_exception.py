class ComponentError(Exception):
    def __init__(self, component_name, component_type):
        Exception.__init__(self)
        self.component_name = component_name
        self.component_type = component_type

    def __str__(self):
        return "[ERROR Component: NAME-{}, TYPE-{}]".format(self.component_name, self.component_type)


class ComponentTypeError(Exception):
    def __init__(self, component_name, component_type):
        Exception.__init__(self)
        self.component_name = component_name
        self.component_type = component_type

    def __str__(self):
        return "[SEFI] ComponentTypeError: ILLEGAL COMPONENT TYPE: '{}' ON {}" \
            .format(self.component_type, self.component_name)


class SubComponentNotFindError(ComponentError):
    def __init__(self, sub_component_name, component_name, component_type):
        ComponentError.__init__(self, component_name, component_type)
        self.sub_component_name = sub_component_name

    def __str__(self):
        return "[SEFI] UserDefinedSubComponentNotFindError: NOT FIND SubComponent-{} \nON {}"\
            .format(self.sub_component_name, ComponentError.__str__(self))


class ComponentNotFindError(ComponentError):
    def __init__(self, component_name, component_type):
        ComponentError.__init__(self, component_name, component_type)

    def __str__(self):
        return "[SEFI] UserDefinedComponentCollectionNotFindError: NOT FIND Component-{} \n" \
               " * The component name defined in the configuration may be incorrect.\n" \
               " * The component may not be registered in the ComponentManager"\
            .format(ComponentError.__str__(self))


class ParamsHandlerComponentTypeError(ComponentError):
    def __init__(self, component_name, component_type, error_type):
        ComponentError.__init__(self, component_name, component_type)
        self.error_type = error_type

    def __str__(self):
        return "[SEFI] ParamsHandlerComponentTypeError: ILLEGAL CONFIG PARAMS HANDLER TYPE: {} \nON {}" \
            .format(self.error_type, ComponentError.__str__(self))


class UtilComponentTypeError(ComponentError):
    def __init__(self, component_name, component_type, error_type):
        ComponentError.__init__(self, component_name, component_type)
        self.error_type = error_type

    def __str__(self):
        return "[SEFI] UtilComponentTypeError: ILLEGAL UTIL TYPE: {} \nON {}" \
            .format(self.error_type, ComponentError.__str__(self))


class ComponentReturnTypeError(ComponentError):
    def __init__(self, sub_component_name, component_name, component_type, need_type, get_type):
        ComponentError.__init__(self, component_name, component_type)
        self.need_type = need_type
        self.get_type = get_type
        self.sub_component_name = sub_component_name

    def __str__(self):
        return "[SEFI] ComponentReturnTypeError: EXPECT RETURN TYPE-{}, BUT GET A TYPE-{} \n" \
               "ON SubComponent-{} \n" \
               "ON {}" \
            .format(self.need_type, self.get_type, self.sub_component_name, ComponentError.__str__(self))
