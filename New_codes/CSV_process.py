from manim import *

class MonolithicArchitecture(Scene):
    def construct(self):
        # Title
        title = Text("Monolithic Architecture", font_size=40).to_edge(UP)
        self.play(Write(title))

        # Explanation
        explanation = Text("Application is built as a single, tightly integrated unit.\nIt is a one-tier architecture.",
                           font_size=25).next_to(title, DOWN)
        self.play(Write(explanation))

        # Diagram
        diagram = self.create_diagram()
        self.play(FadeIn(diagram))

    def create_diagram(self):
        # Clients
        clients_box = RoundedRectangle(width=2, height=1, corner_radius=0.1).shift(LEFT * 3)
        clients_label = Text("Clients", font_size=20).next_to(clients_box, UP)
        clients = VGroup(clients_box, clients_label)

        # Second Subunit
        subunit2 = RoundedRectangle(width=3, height=5, corner_radius=0.2).shift(RIGHT * 0)
        auth_label = Text("Auth", font_size=20).next_to(subunit2.get_top(), DOWN)
        product_label = Text("Product", font_size=20).next_to(subunit2, RIGHT)
        cart_label = Text("Cart", font_size=20).next_to(subunit2.get_bottom(), UP)
        order_label = Text("Order", font_size=20).next_to(subunit2, LEFT)
        subunit2_labels = VGroup(auth_label, product_label, cart_label, order_label)
        subunit2_with_labels = VGroup(subunit2, subunit2_labels)

        # Third Subunit
        subunit3 = RoundedRectangle(width=2, height=5, corner_radius=0.2).shift(RIGHT * 3)
        database_label = Text("Database\nLayer", font_size=20).next_to(subunit3, DOWN)
        subunit3_with_label = VGroup(subunit3, database_label)

        # Connections
        connections = VGroup(
            Arrow(clients.get_right(), subunit2.get_left()),
            Arrow(clients.get_right(), subunit3.get_left()),
            Arrow(subunit2.get_right(), subunit3.get_left())
        )

        # Grouping
        diagram = VGroup(clients, subunit2_with_labels, subunit3_with_label, connections)
        return diagram
